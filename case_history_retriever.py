import streamlit as st
import traceback
from rag_agents import trigger_crew
import openlit

openlit.init(otlp_endpoint="http://127.0.0.1:4318")


def main():
    st.set_page_config(
        page_title="Query Interface",
        page_icon="🤖",
        layout="wide"
    )

    st.title("Medical Case History Retriever")
    st.markdown("Enter your query below to get started.")

    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Query input using a form
    with st.form(key='query_form'):
        query = st.text_input(
            "Enter your query:",
            key="query_input",
            placeholder="Type your query here..."
        )
        submit_button = st.form_submit_button("Submit")

    # Clear history button outside the form
    if st.button("Clear History"):
        st.session_state.chat_history = []

    # Process the query when form is submitted
    if submit_button and query:
        try:
            result = trigger_crew(query)

            # Add to chat history
            st.session_state.chat_history.append({
                "query": query,
                "result": result,
                "error": None
            })

        except Exception as e:
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            st.session_state.chat_history.append({
                "query": query,
                "result": None,
                "error": error_msg
            })

    # Display chat history in reverse chronological order
    st.markdown("### Chat History")
    for item in reversed(st.session_state.chat_history):
        with st.expander(f"Query: {item['query']}", expanded=True):
            if item['result']:
                st.success(item['result'])
            if item['error']:
                st.error(item['error'])

    # Add some helpful information at the bottom
    st.markdown("---")
    st.markdown("""
    **Tips:**
    - Enter your query in the text box above
    - Press Enter or click Submit to process your query
    - Click Clear History to start fresh
    - Each query and its result will be saved in the chat history
    """)


if __name__ == "__main__":
    main()
