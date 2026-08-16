def get_retriever(vectorstore):
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 10,        # تعداد Chunkهایی که برمی‌گرداند
            "fetch_k": 20,  # تعداد Chunkهایی که ابتدا بررسی می‌کند
            "lambda_mult": 0.5
        }
    )

    