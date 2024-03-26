from scrapegraphai.nodes.llm_node import LLMAnswerNode


def main():
    # Definizione dello stato di esempio
    state = {
        "user_input": "Example user prompt",
        "parsed_document": "Example parsed document"
    }

    # Configurazione del modello per il nodo LLMAnswerNode
    model_config = {
        "llm_model": "your_llm_model_name_here"
        # Specifica altre configurazioni del modello se necessario
    }

    # Creazione di un'istanza del nodo LLMAnswerNode
    llm_node = LLMAnswerNode(
        input=["user_input", "parsed_document"],
        output=["answer"],
        model_config=model_config
    )

    # Esecuzione del nodo per ottenere lo stato aggiornato
    updated_state = llm_node.execute(state)

    # Stampa dello stato aggiornato
    print("Updated state:")
    print(updated_state)


if __name__ == "__main__":
    main()
