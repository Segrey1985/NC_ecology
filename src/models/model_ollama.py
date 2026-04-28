from langchain_ollama import ChatOllama


def create_model(model_name):
    # model1 = "qwen3.5:35b"
    # model2 = "nemotron-cascade-2:30b"
    model = ChatOllama(model=model_name, reasoning=False, num_predict=5000)

    return model


if __name__ == "__main__":
    model = create_model("qwen3.5:35b")
    print(model.invoke("Hello World"))
