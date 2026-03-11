#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr uint32_t MOSS_DELAY_DEFAULT_N_VQ = 32;
constexpr llama_token MOSS_DELAY_DEFAULT_PAD_TOKEN_ID = 151643;
constexpr llama_token MOSS_DELAY_DEFAULT_IM_START_TOKEN_ID = 151644;
constexpr llama_token MOSS_DELAY_DEFAULT_IM_END_TOKEN_ID = 151645;
constexpr llama_token MOSS_DELAY_DEFAULT_AUDIO_START_TOKEN_ID = 151652;
constexpr llama_token MOSS_DELAY_DEFAULT_AUDIO_END_TOKEN_ID = 151653;
constexpr llama_token MOSS_DELAY_DEFAULT_AUDIO_USER_SLOT_TOKEN_ID = 151654;
constexpr llama_token MOSS_DELAY_DEFAULT_AUDIO_ASSISTANT_GEN_SLOT_TOKEN_ID = 151656;
constexpr llama_token MOSS_DELAY_DEFAULT_AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID = 151662;
constexpr llama_token MOSS_DELAY_DEFAULT_AUDIO_PAD_CODE = 1024;
constexpr uint32_t MOSS_DELAY_DEFAULT_AUDIO_VOCAB_SIZE = 1024;
constexpr int64_t MOSS_DELAY_INT64_MAX = std::numeric_limits<int64_t>::max();
constexpr float MOSS_NEG_INF = -std::numeric_limits<float>::infinity();

struct moss_sampling_config {
    float text_temperature = 1.5f;
    float text_top_p = 1.0f;
    int32_t text_top_k = 50;
    float audio_temperature = 1.7f;
    float audio_top_p = 0.8f;
    int32_t audio_top_k = 25;
    float audio_repetition_penalty = 1.0f;
};

struct moss_delay_config {
    uint32_t n_vq = MOSS_DELAY_DEFAULT_N_VQ;
    llama_token pad_token_id = MOSS_DELAY_DEFAULT_PAD_TOKEN_ID;
    llama_token im_start_token_id = MOSS_DELAY_DEFAULT_IM_START_TOKEN_ID;
    llama_token im_end_token_id = MOSS_DELAY_DEFAULT_IM_END_TOKEN_ID;
    llama_token audio_start_token_id = MOSS_DELAY_DEFAULT_AUDIO_START_TOKEN_ID;
    llama_token audio_end_token_id = MOSS_DELAY_DEFAULT_AUDIO_END_TOKEN_ID;
    llama_token audio_user_slot_token_id = MOSS_DELAY_DEFAULT_AUDIO_USER_SLOT_TOKEN_ID;
    llama_token audio_assistant_gen_slot_token_id = MOSS_DELAY_DEFAULT_AUDIO_ASSISTANT_GEN_SLOT_TOKEN_ID;
    llama_token audio_assistant_delay_slot_token_id = MOSS_DELAY_DEFAULT_AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID;
    llama_token audio_pad_code = MOSS_DELAY_DEFAULT_AUDIO_PAD_CODE;
    uint32_t audio_vocab_size = MOSS_DELAY_DEFAULT_AUDIO_VOCAB_SIZE;

    size_t packed_stride() const {
        return 1u + n_vq;
    }
};

struct moss_audio_segment {
    std::vector<llama_token> codes;
    size_t n_frames = 0;
};

struct moss_delay_state {
    int32_t audio_length = 0;
    int64_t delayed_length = MOSS_DELAY_INT64_MAX;
    bool is_audio = false;
    bool is_stopping = false;
    int32_t time_step = 0;
    std::vector<llama_token> text_history;

    uint32_t n_vq = MOSS_DELAY_DEFAULT_N_VQ;
    std::vector<llama_token> audio_history;

    size_t audio_frames() const {
        return n_vq == 0 ? 0 : audio_history.size() / n_vq;
    }

    bool empty_audio() const {
        return audio_history.empty();
    }

    const llama_token * audio_frame_ptr(size_t frame_idx) const {
        if (n_vq == 0 || frame_idx >= audio_frames()) {
            return nullptr;
        }
        return audio_history.data() + frame_idx * n_vq;
    }

    void reserve_audio_frames(size_t frames) {
        audio_history.reserve(frames * n_vq);
    }

    void append_audio(const std::vector<llama_token> & frame) {
        GGML_ASSERT(frame.size() == n_vq);
        audio_history.insert(audio_history.end(), frame.begin(), frame.end());
    }

    void append_audio(const llama_token * frame) {
        GGML_ASSERT(frame != nullptr);
        audio_history.insert(audio_history.end(), frame, frame + n_vq);
    }
};

using moss_rng = std::mt19937;

static void print_usage(int argc, char ** argv) {
    (void) argc;
    LOG("\nexample usage:\n");
    LOG("  %s -m model.gguf --print-delay-config\n", argv[0]);
    LOG("\n");
}

static bool parse_meta_i64(const llama_model * model, const char * key, int64_t & out) {
    char buf[128];
    const int32_t n = llama_model_meta_val_str(model, key, buf, sizeof(buf));
    if (n <= 0) {
        return false;
    }

    char * end = nullptr;
    const long long val = std::strtoll(buf, &end, 10);
    if (end == buf || *end != '\0') {
        return false;
    }
    out = val;
    return true;
}

static bool parse_meta_u32(const llama_model * model, const char * key, uint32_t & out) {
    int64_t tmp = 0;
    if (!parse_meta_i64(model, key, tmp) || tmp < 0 || tmp > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    out = static_cast<uint32_t>(tmp);
    return true;
}

static bool parse_meta_token(const llama_model * model, const char * key, llama_token & out) {
    int64_t tmp = 0;
    if (!parse_meta_i64(model, key, tmp) || tmp < std::numeric_limits<llama_token>::min() || tmp > std::numeric_limits<llama_token>::max()) {
        return false;
    }
    out = static_cast<llama_token>(tmp);
    return true;
}

static moss_delay_config moss_delay_config_from_model(const llama_model * model) {
    moss_delay_config cfg;

    parse_meta_u32(model, "moss-tts-delay.n_vq", cfg.n_vq);
    parse_meta_u32(model, "moss-tts-delay.audio_vocab_size", cfg.audio_vocab_size);
    parse_meta_token(model, "moss-tts-delay.audio_pad_code", cfg.audio_pad_code);
    parse_meta_token(model, "moss-tts-delay.pad_token_id", cfg.pad_token_id);
    parse_meta_token(model, "moss-tts-delay.im_start_token_id", cfg.im_start_token_id);
    parse_meta_token(model, "moss-tts-delay.im_end_token_id", cfg.im_end_token_id);
    parse_meta_token(model, "moss-tts-delay.audio_start_token_id", cfg.audio_start_token_id);
    parse_meta_token(model, "moss-tts-delay.audio_end_token_id", cfg.audio_end_token_id);
    parse_meta_token(model, "moss-tts-delay.audio_user_slot_token_id", cfg.audio_user_slot_token_id);
    parse_meta_token(model, "moss-tts-delay.audio_gen_slot_token_id", cfg.audio_assistant_gen_slot_token_id);
    parse_meta_token(model, "moss-tts-delay.audio_delay_slot_token_id", cfg.audio_assistant_delay_slot_token_id);

    return cfg;
}

static size_t moss_audio_vocab_with_pad(const moss_delay_config & cfg) {
    return std::max<size_t>(cfg.audio_vocab_size + 1u, (size_t) cfg.audio_pad_code + 1u);
}

static int64_t moss_find_last_equal(const std::vector<llama_token> & values, llama_token target) {
    for (int64_t i = (int64_t) values.size() - 1; i >= 0; --i) {
        if (values[(size_t) i] == target) {
            return i;
        }
    }
    return -1;
}

static moss_delay_state moss_init_delay_state(
        const std::vector<llama_token> & packed_input_ids,
        const moss_delay_config & cfg) {
    GGML_ASSERT(cfg.n_vq > 0);
    GGML_ASSERT(packed_input_ids.size() % cfg.packed_stride() == 0);

    moss_delay_state state;
    state.n_vq = cfg.n_vq;

    const size_t seq_len = packed_input_ids.size() / cfg.packed_stride();
    state.text_history.resize(seq_len);
    state.reserve_audio_frames(std::max<size_t>(seq_len + 1024, 256));

    for (size_t t = 0; t < seq_len; ++t) {
        const size_t row = t * cfg.packed_stride();
        state.text_history[t] = packed_input_ids[row];
        state.audio_history.insert(
                state.audio_history.end(),
                packed_input_ids.begin() + row + 1,
                packed_input_ids.begin() + row + 1 + cfg.n_vq);
    }

    if (!state.text_history.empty()) {
        const llama_token last_text_token = state.text_history.back();
        const bool is_continuation =
                last_text_token == cfg.audio_start_token_id ||
                last_text_token == cfg.audio_assistant_gen_slot_token_id;
        if (is_continuation) {
            const int64_t audio_start_idx = moss_find_last_equal(state.text_history, cfg.audio_start_token_id);
            if (audio_start_idx >= 0) {
                state.audio_length = (int32_t) (seq_len - (size_t) audio_start_idx);
                state.is_audio = true;
            }
        }
    }

    return state;
}

static void moss_apply_top_p_inplace(std::vector<float> & logits, size_t n_rows, size_t n_vocab, float top_p) {
    if (top_p >= 1.0f) {
        return;
    }

    for (size_t row = 0; row < n_rows; ++row) {
        float max_logit = MOSS_NEG_INF;
        for (size_t col = 0; col < n_vocab; ++col) {
            max_logit = std::max(max_logit, logits[row * n_vocab + col]);
        }

        if (!std::isfinite(max_logit)) {
            continue;
        }

        std::vector<float> probs(n_vocab, 0.0f);
        float sum_exp = 0.0f;
        for (size_t col = 0; col < n_vocab; ++col) {
            const float logit = logits[row * n_vocab + col];
            if (std::isfinite(logit)) {
                probs[col] = std::exp(logit - max_logit);
                sum_exp += probs[col];
            }
        }

        if (!(sum_exp > 0.0f) || !std::isfinite(sum_exp)) {
            continue;
        }

        for (float & p : probs) {
            p /= sum_exp;
        }

        std::vector<size_t> sorted_idx(n_vocab);
        std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
        std::sort(sorted_idx.begin(), sorted_idx.end(), [&](size_t a, size_t b) {
            return probs[a] > probs[b];
        });

        float cum_probs = 0.0f;
        bool prev_remove = false;
        for (size_t rank = 0; rank < n_vocab; ++rank) {
            const size_t idx = sorted_idx[rank];
            cum_probs += probs[idx];

            bool remove = cum_probs > top_p;
            if (rank > 0) {
                remove = prev_remove;
            } else {
                remove = false;
            }
            prev_remove = cum_probs > top_p;

            if (remove) {
                logits[row * n_vocab + idx] = MOSS_NEG_INF;
            }
        }
    }
}

static void moss_apply_repetition_penalty_inplace(
        std::vector<float> & logits,
        size_t n_rows,
        size_t n_vocab,
        const std::vector<llama_token> * prev_tokens,
        float penalty) {
    if (penalty == 1.0f || prev_tokens == nullptr || prev_tokens->empty()) {
        return;
    }

    std::vector<uint8_t> seen(n_vocab, 0);
    for (llama_token tok : *prev_tokens) {
        if (tok >= 0 && (size_t) tok < n_vocab) {
            seen[(size_t) tok] = 1;
        }
    }

    for (size_t col = 0; col < n_vocab; ++col) {
        if (!seen[col]) {
            continue;
        }
        for (size_t row = 0; row < n_rows; ++row) {
            float & logit = logits[row * n_vocab + col];
            if (logit > 0.0f) {
                logit /= penalty;
            } else {
                logit *= penalty;
            }
        }
    }
}

static llama_token moss_argmax_row(const std::vector<float> & logits, size_t row, size_t n_vocab) {
    size_t best_idx = 0;
    float best_val = logits[row * n_vocab + 0];
    for (size_t col = 1; col < n_vocab; ++col) {
        const float cur = logits[row * n_vocab + col];
        if (cur > best_val) {
            best_val = cur;
            best_idx = col;
        }
    }
    return (llama_token) best_idx;
}

static llama_token moss_multinomial_row(
        const std::vector<float> & probs,
        size_t row,
        size_t n_vocab,
        moss_rng & rng) {
    const float * row_probs = probs.data() + row * n_vocab;
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    const float r = dist(rng);

    float cum = 0.0f;
    size_t idx = 0;
    for (; idx < n_vocab; ++idx) {
        cum += row_probs[idx];
        if (!(cum < r)) {
            break;
        }
    }
    if (idx >= n_vocab) {
        idx = n_vocab - 1;
    }
    return (llama_token) idx;
}

static std::vector<float> moss_softmax(const std::vector<float> & logits, size_t n_rows, size_t n_vocab) {
    std::vector<float> probs(n_rows * n_vocab, 0.0f);

    for (size_t row = 0; row < n_rows; ++row) {
        float max_logit = MOSS_NEG_INF;
        for (size_t col = 0; col < n_vocab; ++col) {
            max_logit = std::max(max_logit, logits[row * n_vocab + col]);
        }

        if (!std::isfinite(max_logit)) {
            probs[row * n_vocab + 0] = 1.0f;
            continue;
        }

        float sum_exp = 0.0f;
        for (size_t col = 0; col < n_vocab; ++col) {
            const float logit = logits[row * n_vocab + col];
            if (std::isfinite(logit)) {
                probs[row * n_vocab + col] = std::exp(logit - max_logit);
                sum_exp += probs[row * n_vocab + col];
            }
        }

        if (!(sum_exp > 0.0f) || !std::isfinite(sum_exp)) {
            probs[row * n_vocab + 0] = 1.0f;
            continue;
        }

        for (size_t col = 0; col < n_vocab; ++col) {
            probs[row * n_vocab + col] /= sum_exp;
        }
    }

    return probs;
}

static std::vector<llama_token> moss_sample_token(
        const std::vector<float> & logits_in,
        size_t n_rows,
        size_t n_vocab,
        moss_rng & rng,
        const std::vector<llama_token> * prev_tokens = nullptr,
        float repetition_penalty = 1.0f,
        float top_p = 1.0f,
        int32_t top_k = 0,
        bool do_sample = true) {
    GGML_ASSERT(logits_in.size() == n_rows * n_vocab);

    std::vector<float> logits = logits_in;
    moss_apply_repetition_penalty_inplace(logits, n_rows, n_vocab, prev_tokens, repetition_penalty);

    std::vector<llama_token> tokens(n_rows, 0);
    if (!do_sample) {
        for (size_t row = 0; row < n_rows; ++row) {
            tokens[row] = moss_argmax_row(logits, row, n_vocab);
        }
        return tokens;
    }

    if (top_k > 0) {
        const size_t k = std::min<size_t>((size_t) top_k, n_vocab);
        for (size_t row = 0; row < n_rows; ++row) {
            std::vector<size_t> top_idx(n_vocab);
            std::iota(top_idx.begin(), top_idx.end(), 0);
            std::nth_element(top_idx.begin(), top_idx.end() - k, top_idx.end(), [&](size_t a, size_t b) {
                return logits[row * n_vocab + a] < logits[row * n_vocab + b];
            });
            top_idx.erase(top_idx.begin(), top_idx.end() - k);

            std::vector<float> top_vals(k);
            for (size_t i = 0; i < k; ++i) {
                top_vals[i] = logits[row * n_vocab + top_idx[i]];
            }

            if (top_p < 1.0f) {
                moss_apply_top_p_inplace(top_vals, 1, k, top_p);
            }

            const std::vector<float> probs = moss_softmax(top_vals, 1, k);
            const llama_token local = moss_multinomial_row(probs, 0, k, rng);
            tokens[row] = (llama_token) top_idx[(size_t) local];
        }
        return tokens;
    }

    if (top_p < 1.0f) {
        moss_apply_top_p_inplace(logits, n_rows, n_vocab, top_p);
    }
    const std::vector<float> probs = moss_softmax(logits, n_rows, n_vocab);
    for (size_t row = 0; row < n_rows; ++row) {
        tokens[row] = moss_multinomial_row(probs, row, n_vocab, rng);
    }

    return tokens;
}

static std::vector<llama_token> moss_collect_audio_history_channels(
        const moss_delay_state & state,
        const std::vector<size_t> & channels) {
    if (channels.empty() || state.empty_audio()) {
        return {};
    }

    std::vector<llama_token> out;
    out.reserve(state.audio_frames() * channels.size());
    for (size_t frame = 0; frame < state.audio_frames(); ++frame) {
        const llama_token * audio = state.audio_frame_ptr(frame);
        for (size_t channel : channels) {
            out.push_back(audio[channel]);
        }
    }
    return out;
}

static std::vector<llama_token> moss_delay_step(
        moss_delay_state & state,
        const std::vector<float> & text_logits,
        const std::vector<float> & audio_logits,
        const moss_sampling_config & sampling_cfg,
        const moss_delay_config & cfg,
        moss_rng & rng) {
    GGML_ASSERT(cfg.n_vq == state.n_vq);

    const size_t n_vq = cfg.n_vq;
    const size_t text_vocab = text_logits.size();
    const size_t audio_vocab = moss_audio_vocab_with_pad(cfg);
    GGML_ASSERT(audio_logits.size() == n_vq * audio_vocab);

    std::vector<llama_token> result(cfg.packed_stride(), cfg.audio_pad_code);
    if (state.is_stopping) {
        result[0] = cfg.pad_token_id;
        return result;
    }

    llama_token next_text = cfg.pad_token_id;

    if (state.delayed_length < (int64_t) n_vq) {
        next_text = cfg.audio_assistant_delay_slot_token_id;
    } else if (state.delayed_length == (int64_t) n_vq) {
        next_text = cfg.audio_end_token_id;
        state.is_audio = false;
    } else {
        std::vector<float> scaled = text_logits;
        const float text_temp = sampling_cfg.text_temperature > 0.0f ? sampling_cfg.text_temperature : 1.0f;
        const bool text_do_sample = sampling_cfg.text_temperature > 0.0f;
        for (float & v : scaled) {
            v /= text_temp;
        }

        if (!state.is_audio) {
            const llama_token excluded[] = {
                cfg.pad_token_id,
                cfg.audio_assistant_gen_slot_token_id,
                cfg.audio_assistant_delay_slot_token_id,
                cfg.audio_end_token_id,
            };
            for (llama_token tok : excluded) {
                if (tok >= 0 && (size_t) tok < text_vocab) {
                    scaled[(size_t) tok] = MOSS_NEG_INF;
                }
            }
        } else {
            std::fill(scaled.begin(), scaled.end(), MOSS_NEG_INF);
            if ((size_t) cfg.audio_assistant_gen_slot_token_id < text_vocab) {
                scaled[(size_t) cfg.audio_assistant_gen_slot_token_id] =
                        text_logits[(size_t) cfg.audio_assistant_gen_slot_token_id] / text_temp;
            }
            if ((size_t) cfg.audio_assistant_delay_slot_token_id < text_vocab) {
                scaled[(size_t) cfg.audio_assistant_delay_slot_token_id] =
                        text_logits[(size_t) cfg.audio_assistant_delay_slot_token_id] / text_temp;
            }
        }

        if (state.time_step == 0 && (size_t) cfg.audio_assistant_delay_slot_token_id < text_vocab) {
            scaled[(size_t) cfg.audio_assistant_delay_slot_token_id] = MOSS_NEG_INF;
        }
        if (state.time_step <= (int32_t) n_vq && (size_t) cfg.im_end_token_id < text_vocab) {
            scaled[(size_t) cfg.im_end_token_id] = MOSS_NEG_INF;
        }

        next_text = moss_sample_token(
                scaled, 1, text_vocab, rng, nullptr, 1.0f,
                sampling_cfg.text_top_p, sampling_cfg.text_top_k, text_do_sample)[0];
    }

    if (next_text == cfg.audio_start_token_id) {
        state.is_audio = true;
    }
    if (next_text == cfg.im_end_token_id) {
        state.is_stopping = true;
    }

    std::vector<llama_token> next_audio(n_vq, cfg.audio_pad_code);
    bool any_sampling = false;
    for (size_t channel = 0; channel < n_vq; ++channel) {
        const bool pre_audio = channel < (size_t) std::max(state.audio_length, 0);
        const bool post_audio = state.delayed_length == MOSS_DELAY_INT64_MAX ||
                channel > (size_t) std::max<int64_t>(state.delayed_length - 1, -1);
        any_sampling = any_sampling || (pre_audio && post_audio);
    }

    if (any_sampling) {
        std::vector<float> scaled_audio = audio_logits;
        const float audio_temp = sampling_cfg.audio_temperature > 0.0f ? sampling_cfg.audio_temperature : 1.0f;
        const bool audio_do_sample = sampling_cfg.audio_temperature > 0.0f;
        for (float & v : scaled_audio) {
            v /= audio_temp;
        }
        if ((size_t) cfg.audio_pad_code < audio_vocab) {
            for (size_t channel = 0; channel < n_vq; ++channel) {
                scaled_audio[channel * audio_vocab + (size_t) cfg.audio_pad_code] = MOSS_NEG_INF;
            }
        }

        const bool sample_ch0 =
                0 < (size_t) std::max(state.audio_length, 0) &&
                (state.delayed_length == MOSS_DELAY_INT64_MAX ||
                 0 > std::max<int64_t>(state.delayed_length - 1, -1));
        if (sample_ch0) {
            const std::vector<size_t> ch0 = {0};
            const std::vector<llama_token> prev = moss_collect_audio_history_channels(state, ch0);
            const std::vector<float> ch0_logits(scaled_audio.begin(), scaled_audio.begin() + audio_vocab);
            next_audio[0] = moss_sample_token(
                    ch0_logits, 1, audio_vocab, rng, &prev,
                    sampling_cfg.audio_repetition_penalty,
                    sampling_cfg.audio_top_p,
                    sampling_cfg.audio_top_k,
                    audio_do_sample)[0];
        }

        std::vector<size_t> rest_channels;
        for (size_t channel = 1; channel < n_vq; ++channel) {
            const bool pre_audio = channel < (size_t) std::max(state.audio_length, 0);
            const bool post_audio = state.delayed_length == MOSS_DELAY_INT64_MAX ||
                    channel > (size_t) std::max<int64_t>(state.delayed_length - 1, -1);
            if (pre_audio && post_audio) {
                rest_channels.push_back(channel);
            }
        }

        if (!rest_channels.empty()) {
            std::vector<float> rest_logits(rest_channels.size() * audio_vocab);
            for (size_t i = 0; i < rest_channels.size(); ++i) {
                const size_t channel = rest_channels[i];
                std::copy_n(
                        scaled_audio.begin() + channel * audio_vocab,
                        audio_vocab,
                        rest_logits.begin() + i * audio_vocab);
            }
            const std::vector<llama_token> prev = moss_collect_audio_history_channels(state, rest_channels);
            const std::vector<llama_token> sampled = moss_sample_token(
                    rest_logits, rest_channels.size(), audio_vocab, rng, &prev,
                    sampling_cfg.audio_repetition_penalty,
                    sampling_cfg.audio_top_p,
                    sampling_cfg.audio_top_k,
                    audio_do_sample);
            for (size_t i = 0; i < rest_channels.size(); ++i) {
                next_audio[rest_channels[i]] = sampled[i];
            }
        }
    }

    if (next_text == cfg.audio_start_token_id ||
            next_text == cfg.audio_assistant_gen_slot_token_id ||
            next_text == cfg.audio_assistant_delay_slot_token_id) {
        state.audio_length += 1;
    }
    if (next_text == cfg.audio_end_token_id) {
        state.audio_length = 0;
    }

    if (state.delayed_length == MOSS_DELAY_INT64_MAX && next_text == cfg.audio_assistant_delay_slot_token_id) {
        state.delayed_length = 0;
    }
    if (state.delayed_length != MOSS_DELAY_INT64_MAX) {
        state.delayed_length += 1;
    }
    if (state.delayed_length > (int64_t) n_vq) {
        state.delayed_length = MOSS_DELAY_INT64_MAX;
    }

    state.time_step += 1;
    state.text_history.push_back(next_text);
    state.append_audio(next_audio);

    result[0] = next_text;
    std::copy(next_audio.begin(), next_audio.end(), result.begin() + 1);
    return result;
}

static std::vector<llama_token> moss_apply_delay_pattern(
        const std::vector<llama_token> & codes,
        size_t n_frames,
        const moss_delay_config & cfg) {
    GGML_ASSERT(cfg.n_vq > 0);
    GGML_ASSERT(codes.size() == n_frames * cfg.n_vq);

    const size_t delayed_frames = n_frames + cfg.n_vq - 1;
    std::vector<llama_token> delayed(delayed_frames * cfg.n_vq, cfg.audio_pad_code);

    for (size_t channel = 0; channel < cfg.n_vq; ++channel) {
        for (size_t t = 0; t < n_frames; ++t) {
            delayed[(channel + t) * cfg.n_vq + channel] = codes[t * cfg.n_vq + channel];
        }
    }

    return delayed;
}

static std::vector<llama_token> moss_apply_de_delay_pattern(
        const std::vector<llama_token> & delayed_codes,
        size_t delayed_frames,
        const moss_delay_config & cfg,
        size_t * out_frames = nullptr) {
    GGML_ASSERT(cfg.n_vq > 0);
    GGML_ASSERT(delayed_codes.size() == delayed_frames * cfg.n_vq);

    if (delayed_frames + 1 <= cfg.n_vq) {
        if (out_frames != nullptr) {
            *out_frames = 0;
        }
        return {};
    }

    const size_t n_frames = delayed_frames - cfg.n_vq + 1;
    std::vector<llama_token> codes(n_frames * cfg.n_vq);
    for (size_t channel = 0; channel < cfg.n_vq; ++channel) {
        for (size_t t = 0; t < n_frames; ++t) {
            codes[t * cfg.n_vq + channel] = delayed_codes[(channel + t) * cfg.n_vq + channel];
        }
    }

    if (out_frames != nullptr) {
        *out_frames = n_frames;
    }

    return codes;
}

static std::vector<moss_audio_segment> moss_extract_audio_segments(
        const std::vector<llama_token> & generation_audio,
        size_t delayed_frames,
        const moss_delay_config & cfg) {
    size_t n_frames = 0;
    const std::vector<llama_token> codes = moss_apply_de_delay_pattern(generation_audio, delayed_frames, cfg, &n_frames);
    if (n_frames == 0) {
        return {};
    }

    std::vector<moss_audio_segment> segments;
    size_t cur_start = SIZE_MAX;

    for (size_t t = 0; t < n_frames; ++t) {
        bool is_pad = true;
        for (size_t channel = 0; channel < cfg.n_vq; ++channel) {
            if (codes[t * cfg.n_vq + channel] != cfg.audio_pad_code) {
                is_pad = false;
                break;
            }
        }

        if (!is_pad && cur_start == SIZE_MAX) {
            cur_start = t;
        }

        const bool close_segment = cur_start != SIZE_MAX && (is_pad || t + 1 == n_frames);
        if (close_segment) {
            const size_t cur_end = is_pad ? t : t + 1;
            moss_audio_segment seg;
            seg.n_frames = cur_end - cur_start;
            seg.codes.insert(
                    seg.codes.end(),
                    codes.begin() + cur_start * cfg.n_vq,
                    codes.begin() + cur_end * cfg.n_vq);
            segments.push_back(std::move(seg));
            cur_start = SIZE_MAX;
        }
    }

    return segments;
}

static std::string moss_delay_config_to_string(const moss_delay_config & cfg) {
    std::ostringstream oss;
    oss
        << "n_vq=" << cfg.n_vq
        << " pad_token_id=" << cfg.pad_token_id
        << " im_start_token_id=" << cfg.im_start_token_id
        << " im_end_token_id=" << cfg.im_end_token_id
        << " audio_start_token_id=" << cfg.audio_start_token_id
        << " audio_end_token_id=" << cfg.audio_end_token_id
        << " audio_user_slot_token_id=" << cfg.audio_user_slot_token_id
        << " audio_gen_slot_token_id=" << cfg.audio_assistant_gen_slot_token_id
        << " audio_delay_slot_token_id=" << cfg.audio_assistant_delay_slot_token_id
        << " audio_pad_code=" << cfg.audio_pad_code
        << " audio_vocab_size=" << cfg.audio_vocab_size;
    return oss.str();
}

static bool moss_delay_self_test() {
    moss_delay_config cfg;

    std::vector<llama_token> codes = {
        10, 11, 12,
        20, 21, 22,
        30, 31, 32,
    };
    cfg.n_vq = 3;
    cfg.audio_pad_code = 99;

    const std::vector<llama_token> delayed = moss_apply_delay_pattern(codes, 3, cfg);
    const std::vector<llama_token> expected_delayed = {
        10, 99, 99,
        20, 11, 99,
        30, 21, 12,
        99, 31, 22,
        99, 99, 32,
    };
    if (delayed != expected_delayed) {
        return false;
    }

    size_t dedelayed_frames = 0;
    const std::vector<llama_token> restored = moss_apply_de_delay_pattern(delayed, 5, cfg, &dedelayed_frames);
    if (dedelayed_frames != 3 || restored != codes) {
        return false;
    }

    std::vector<llama_token> packed = {
        1, 99, 99, 99,
        cfg.audio_start_token_id, 10, 11, 12,
        cfg.audio_assistant_gen_slot_token_id, 20, 21, 22,
    };
    const moss_delay_state state = moss_init_delay_state(packed, cfg);
    if (!(state.text_history.size() == 3 &&
            state.audio_frames() == 3 &&
            state.is_audio &&
            state.audio_length == 2 &&
            !state.is_stopping &&
            state.time_step == 0)) {
        return false;
    }

    {
        std::vector<float> logits = {
            3.0f, 2.0f, 1.0f,
            1.0f, 3.0f, 2.0f,
        };
        std::vector<llama_token> prev = {1};
        moss_apply_repetition_penalty_inplace(logits, 2, 3, &prev, 2.0f);
        if (std::fabs(logits[1] - 1.0f) > 1e-6f || std::fabs(logits[4] - 1.5f) > 1e-6f) {
            return false;
        }
    }

    {
        std::vector<float> logits = {5.0f, 4.0f, 1.0f};
        moss_apply_top_p_inplace(logits, 1, 3, 0.7f);
        if (!std::isfinite(logits[0]) || std::isfinite(logits[1]) || std::isfinite(logits[2])) {
            return false;
        }
    }

    {
        moss_rng rng(123);
        const std::vector<float> logits = {
            1.0f, 9.0f, 3.0f,
            2.0f, 1.0f, 8.0f,
        };
        const std::vector<llama_token> sampled = moss_sample_token(logits, 2, 3, rng, nullptr, 1.0f, 1.0f, 1, true);
        if (sampled.size() != 2 || sampled[0] != 1 || sampled[1] != 2) {
            return false;
        }
    }

    {
        moss_delay_state step_state;
        step_state.n_vq = 3;
        step_state.audio_length = 2;
        step_state.is_audio = true;
        step_state.text_history = {cfg.audio_start_token_id, cfg.audio_assistant_gen_slot_token_id};
        step_state.audio_history = {
            3, 4, cfg.audio_pad_code,
            5, 6, cfg.audio_pad_code,
        };

        const std::vector<float> text_logits = {
            0.0f, 0.0f, 0.0f, 0.0f, 10.0f, 9.0f, 0.0f, 0.0f,
        };
        moss_delay_config step_cfg = cfg;
        step_cfg.pad_token_id = 0;
        step_cfg.im_end_token_id = 1;
        step_cfg.audio_start_token_id = 2;
        step_cfg.audio_end_token_id = 3;
        step_cfg.audio_assistant_gen_slot_token_id = 4;
        step_cfg.audio_assistant_delay_slot_token_id = 5;
        step_cfg.audio_pad_code = 7;
        step_cfg.audio_vocab_size = 7;

        const std::vector<float> audio_logits = {
            1.0f, 8.0f, 2.0f, 1.0f, 1.0f, 1.0f, 1.0f, -100.0f,
            2.0f, 1.0f, 9.0f, 1.0f, 1.0f, 1.0f, 1.0f, -100.0f,
            9.0f, 1.0f, 2.0f, 1.0f, 1.0f, 1.0f, 1.0f, -100.0f,
        };
        moss_sampling_config sampling_cfg;
        sampling_cfg.text_temperature = 1.0f;
        sampling_cfg.text_top_k = 1;
        sampling_cfg.audio_temperature = 1.0f;
        sampling_cfg.audio_top_k = 1;

        moss_rng rng(7);
        const std::vector<llama_token> next = moss_delay_step(
                step_state, text_logits, audio_logits, sampling_cfg, step_cfg, rng);
        if (next.size() != 4 || next[0] != 4 || next[1] != 1 || next[2] != 2 || next[3] != 7) {
            return false;
        }
    }

    return true;
}

} // namespace

int main(int argc, char ** argv) {
    std::string model_path;
    bool print_delay_config = false;
    bool self_test = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            model_path = argv[++i];
            continue;
        }
        if (arg == "--print-delay-config") {
            print_delay_config = true;
            continue;
        }
        if (arg == "--self-test-delay-state") {
            self_test = true;
            continue;
        }
        if (arg == "-h" || arg == "--help") {
            print_usage(argc, argv);
            return EXIT_SUCCESS;
        }

        LOG_ERR("unknown argument: %s\n", arg.c_str());
        print_usage(argc, argv);
        return EXIT_FAILURE;
    }

    if (self_test) {
        if (!moss_delay_self_test()) {
            LOG_ERR("moss delay state self-test failed\n");
            return EXIT_FAILURE;
        }
        LOG("moss delay state self-test: ok\n");
    }

    if (!print_delay_config) {
        if (self_test) {
            return EXIT_SUCCESS;
        }
        LOG("moss delay state and multi-head sampler are in place; audio decode is not implemented yet.\n");
        LOG("use --print-delay-config with -m <model.gguf> to inspect model metadata.\n");
        return EXIT_SUCCESS;
    }

    if (model_path.empty()) {
        LOG_ERR("--print-delay-config requires -m <model.gguf>\n");
        return EXIT_FAILURE;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.use_mmap = true;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (model == nullptr) {
        LOG_ERR("failed to load model: %s\n", model_path.c_str());
        llama_backend_free();
        return EXIT_FAILURE;
    }

    const moss_delay_config cfg = moss_delay_config_from_model(model);
    LOG("%s\n", moss_delay_config_to_string(cfg).c_str());

    llama_model_free(model);
    llama_backend_free();

    return EXIT_SUCCESS;
}
