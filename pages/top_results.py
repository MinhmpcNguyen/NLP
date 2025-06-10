import streamlit as st

st.set_page_config(page_title="Top Results - HUST Assistant", initial_sidebar_state="collapsed")

st.title("Data Resource")
    
top_results = st.session_state.get("top_results", [])

if not top_results:
    st.warning("No available result.")
else:
    for idx, result in enumerate(top_results, 1):
        st.markdown(f"[{result['url']}]({result['url']})  \n{result['data']}")
        st.markdown("---")

if st.button("Get back to chat page"):
    st.switch_page("ui.py")
