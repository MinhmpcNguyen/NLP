import streamlit as st

st.set_page_config(
    page_title="HUST Assistant",
    initial_sidebar_state="expanded",
)

st.title("Data Resource")
    
top_results = st.session_state.get("top_results")

if not top_results:
    st.warning("No available result.")
else:
    for idx, result in enumerate(top_results, 1):
        st.markdown(f"""
        <div style="background-color: #f8f8f8; padding: 15px; border-radius: 10px; border: 1px solid #ddd; color: #000;">
            <p style="margin: 0; color: #000;"><b>URL {idx}:</b> <a href="{result['url']}" target="_blank" style="color: #0066cc;">{result['url']}</a></p>
            <p style="margin: 5px 0 0 0; color: #000;"><b>Data:</b> {result['data']}</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")

st.markdown("---")

if st.button("Back to Chat"):
    st.switch_page(r"pages\chat.py")