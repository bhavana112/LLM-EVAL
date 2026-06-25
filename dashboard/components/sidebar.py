import streamlit as st
from dashboard.utils import check_backend_health

PAGES = [
    "🏠 Home",
    "🧪 Experiments",
    "📄 Reports",
    "📊 Comparisons",
    "⚠️ Regression Detection",
    "🤖 Failure Analysis"
]

def render_sidebar() -> str:
    """
    Renders the sidebar navigation, title, and connection status indicator.
    Returns the selected page name.
    """
    with st.sidebar:
        st.title("🎯 LLM Evaluation Platform")
        st.subheader("Visualization & Analytics Layer")
        st.markdown("---")
        
        # Navigation
        st.subheader("Navigation")
        selected_page = st.radio(
            label="Go to Page:",
            options=PAGES,
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Backend health indicator
        st.subheader("System Status")
        is_online = check_backend_health()
        
        if is_online:
            st.success("● Backend API: ONLINE")
        else:
            st.warning("● Backend API: OFFLINE (Local Mock Mode)")
            
        st.caption("Default Port: http://localhost:8000")
        
    return selected_page
