import Foundation

// import llama

enum LlamaError: Error {
    case couldNotInitializeContext
}

actor LlamaContext {
    /// model is the pointer to model weight?
    private var model: OpaquePointer
    /// context is the pointer to information for the context state
    private var context: OpaquePointer
    /// batch is the unit of inference
    private var batch: llama_batch
    /// current token list
    private var tokens_list: [llama_token]
    /// generated cchars list
    private var cchars_list: [CChar]

    var n_len: Int32 = 2048
    var n_cur: Int32 = 0
    var n_decode: Int32 = 0

    init(model: OpaquePointer, context: OpaquePointer) {
        self.model = model
        self.context = context
        self.tokens_list = []
        self.batch = llama_batch_init(n_len, 0, 1)
        self.cchars_list = []
    }

    deinit {
        llama_free(context)
        llama_free_model(model)
        llama_backend_free()
    }

    /// create new instance
    static func createContext(path: String) throws -> LlamaContext {
        llama_backend_init(false)
        // パラメータを取得
        let model_params = llama_model_default_params()

        // モデルを読み込む
        let model = llama_load_model_from_file(path, model_params)
        guard let model else {
            print("Could not load model at \(path)")
            throw LlamaError.couldNotInitializeContext
        }
        // 実行情報
        var ctx_params = llama_context_default_params()
        ctx_params.seed = 1234
        ctx_params.n_ctx = 2048
        ctx_params.n_threads = 8
        ctx_params.n_threads_batch = 8
        // コンテキストを生成
        let context = llama_new_context_with_model(model, ctx_params)
        guard let context else {
            print("Could not load context!")
            throw LlamaError.couldNotInitializeContext
        }
        
        return LlamaContext(model: model, context: context)
    }

    /// 初回に呼ばれる
    func completion_init(text: String) {
        print("attempting to complete \"\(text)\"")
        // トークナイズを実行し、リストを初期化
        tokens_list = tokenize(text: text, add_bos: true)
        cchars_list = text.cString(using: .utf8) ?? []
        if cchars_list.last == .zero {
            cchars_list.removeLast()
        }
        // ctxの数
        let n_ctx = llama_n_ctx(context)
        // 長さ
        let n_kv_req = n_len

        print("\n n_len = \(n_len), n_ctx = \(n_ctx), n_kv_req = \(n_kv_req)")

        if n_kv_req > n_ctx {
            print("error: n_kv_req > n_ctx, the required KV cache size is not big enough")
        }

        for id in tokens_list {
            print(token_to_piece(token: id) as [CChar], token_to_piece(token: id) as String)
        }

        // batch = llama_batch_init(512, 0) // done in init()
        batch.n_tokens = Int32(tokens_list.count)

        for i1 in 0..<batch.n_tokens {
            let i = Int(i1)
            batch.token[i] = tokens_list[i]
            batch.pos[i] = i1
            batch.n_seq_id[Int(i)] = 1
            batch.seq_id[Int(i)]![0] = 0
            batch.logits[i] = 0
        }
        batch.logits[Int(batch.n_tokens) - 1] = 1 // true

        if llama_decode(context, batch) != 0 {
            print("llama_decode() failed")
        }

        n_cur = batch.n_tokens
    }

    func completion_loop() -> String {
        var new_token_id: llama_token = 0

        let n_vocab = llama_n_vocab(model)
        let logits = llama_get_logits_ith(context, batch.n_tokens - 1)

        var candidates = Array<llama_token_data>()
        candidates.reserveCapacity(Int(n_vocab))

        for token_id in 0..<n_vocab {
            candidates.append(llama_token_data(id: token_id, logit: logits![Int(token_id)], p: 0.0))
        }
        candidates.withUnsafeMutableBufferPointer() { buffer in
            var candidates_p = llama_token_data_array(data: buffer.baseAddress, size: buffer.count, sorted: false)
//            llama_sample_repetition_penalties(context, &candidates_p, tokens_list, 64, 1.1, 0, 0)
            llama_sample_top_k(context, &candidates_p, 40, 1)
            llama_sample_top_p(context, &candidates_p, 0.95, 1)
            llama_sample_min_p(context, &candidates_p, 0.05, 1)
            llama_sample_temp(context, &candidates_p, 0.8)
//            new_token_id = llama_sample_token_greedy(context, &candidates_p)
            new_token_id = llama_sample_token(context, &candidates_p)
        }

        if new_token_id == llama_token_eos(context) || n_cur == n_len {
            print("\n")
            return String(cString: cchars_list + [0])
        }

        let new_token_cchars: [CChar] = token_to_piece(token: new_token_id)
        print(cchars_list.count, token_to_piece(token: new_token_id) as String, new_token_cchars)
        cchars_list.append(contentsOf: new_token_cchars.dropLast())

        batch.n_tokens = 0

        batch.token[Int(batch.n_tokens)] = new_token_id
        batch.pos[Int(batch.n_tokens)] = n_cur
        batch.n_seq_id[Int(batch.n_tokens)] = 1
        batch.seq_id[Int(batch.n_tokens)]![0] = 0
        batch.logits[Int(batch.n_tokens)] = 1 // true
        batch.n_tokens += 1

        n_decode += 1

        n_cur += 1

        if llama_decode(context, batch) != 0 {
            print("failed to evaluate llama!")
        }

        return String(cString: cchars_list + [0])
    }

    func clear() {
        tokens_list.removeAll()
    }

    private func tokenize(text: String, add_bos: Bool) -> [llama_token] {
        let n_tokens = text.utf8.count + (add_bos ? 1 : 0)
        let tokens = UnsafeMutablePointer<llama_token>.allocate(capacity: n_tokens)
        let tokenCount = llama_tokenize(model, text, Int32(text.utf8.count), tokens, Int32(n_tokens), add_bos, false)

        var swiftTokens: [llama_token] = []
        for i in 0..<tokenCount {
            swiftTokens.append(tokens[Int(i)])
        }

        tokens.deallocate()

        return swiftTokens
    }

    private func token_to_piece(token: llama_token) -> [CChar] {
        let result = UnsafeMutablePointer<Int8>.allocate(capacity: 8)
        result.initialize(repeating: Int8(0), count: 8)
        defer {
            result.deallocate()
        }
        let nTokens = llama_token_to_piece(model, token, result, 8)

        if nTokens < 0 {
            let newResult = UnsafeMutablePointer<Int8>.allocate(capacity: Int(-nTokens))
            newResult.initialize(repeating: Int8(0), count: Int(-nTokens))
            defer {
                newResult.deallocate()
            }
            let nTokens = llama_token_to_piece(model, token, newResult, -nTokens)
            let bufferPointer = UnsafeBufferPointer(start: newResult, count: Int(nTokens))
            return Array(bufferPointer) + [0]
        } else {
            let bufferPointer = UnsafeBufferPointer(start: result, count: Int(nTokens))
            return Array(bufferPointer) + [0]
        }
    }

    private func token_to_piece(token: llama_token) -> String {
        let cchars: [CChar] = token_to_piece(token: token)
        return String(cString: cchars)
    }
}
