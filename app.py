#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Protein RAG system Streamlit interface
Provides a user-friendly web UI for protein sequence analysis
"""

import streamlit as st
import pandas as pd
import random
import string
import time
from main import get_protein_service, initialize_service

# Page configuration
st.set_page_config(
    page_title="Protein RAG Retrieval System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state initialization
if 'upload_success' not in st.session_state:
    st.session_state.upload_success = False
if 'success_message' not in st.session_state:
    st.session_state.success_message = ""
if 'upload_stats' not in st.session_state:
    st.session_state.upload_stats = {}
if 'should_refresh_stats' not in st.session_state:
    st.session_state.should_refresh_stats = False
if 'last_stats_refresh' not in st.session_state:
    st.session_state.last_stats_refresh = time.time()

# Page title
st.title("🧬 Protein Sequence Retrieval & Analysis System")
st.markdown("ESM2-based protein similarity search platform")

# Initialize service
@st.cache_resource(show_spinner=False)
def init_service():
    """Initialize protein service"""
    service = get_protein_service()
    if not service._db_connected:
        service.connect_database()
    return service._db_connected

service_ready = init_service()

# Real-time stats (no cache)
def get_real_time_stats():
    """Get real-time database statistics"""
    service = get_protein_service()
    if not service._db_connected:
        service.connect_database()
    connection_status = service.check_database_connection()
    if connection_status["db_connected"]:
        stats = service.get_collection_stats()
        st.session_state.last_stats_refresh = time.time()
        return stats
    return {"total_proteins": 0, "collection_name": "protein_collection", "is_loaded": False}

# Force refresh
def refresh_stats():
    """Force refresh statistics and clear caches"""
    if hasattr(st, 'cache_data'):
        st.cache_data.clear()
    st.session_state.should_refresh_stats = True
    service = get_protein_service()
    if not service._db_connected:
        service.connect_database()

# Sidebar - System status
with st.sidebar:
    st.header("🔧 System Status")

    with st.spinner("Connecting to database..."):
        service = get_protein_service()
        if not service._db_connected or st.session_state.should_refresh_stats:
            service.connect_database()
            st.session_state.should_refresh_stats = False

    connection_status = service.check_database_connection()

    if connection_status["db_connected"]:
        st.success("✅ System & database ready")
        stats = get_real_time_stats()
        st.metric(
            "Protein count in DB",
            stats.get("total_proteins", 0)
        )
        st.info(f"Collection name: {stats.get('collection_name', 'N/A')}")
        st.success("📊 Database status: Connected")
        if stats.get("total_proteins", 0) == 0:
            st.warning("💡 Database empty, please upload FASTA file")
        else:
            st.success(f"🗄️ Database contains {stats.get('total_proteins', 0)} protein records")
        if st.button("🔄 Refresh Stats", use_container_width=True, help="Manually refresh database stats"):
            refresh_stats()
            st.rerun()
    else:
        st.error("📊 Database status: Connection failed")
        if st.button("🔄 Retry Connection", use_container_width=True):
            service.connect_database()
            st.rerun()

    if connection_status["model_loaded"]:
        st.success("🤖 ESM2 Model: Loaded")
    else:
        st.info("🤖 ESM2 Model: Not loaded")
        st.info("💡 Model will auto-load on first protein processing")

    with st.expander("ℹ️ System Information"):
        st.markdown(
        """
        **🚀 Features:**
        - Milvus Lite lightweight vector database
        - Local file storage, no server needed
        - Auto-load ESM2 model on first use
        - Automatic index management
        
        **🧬 Supported Formats:**
        - FASTA (.fasta, .fa, .txt)
        - Standard single-letter amino acid codes
        
        **🔍 Search:**
        - ESM2 embeddings
        - AUTOINDEX (Milvus Lite compatible)
        - L2 distance metric
        """
        )

    st.markdown("---")
    with st.expander("⚠️ Danger Zone", expanded=False):
        st.markdown("### 🗑️ Clear Database")
        st.warning("Warning: This will permanently delete all protein data and cannot be undone.")
        if connection_status["db_connected"]:
            stats = service.get_collection_stats()
            if stats.get("total_proteins", 0) > 0:
                if st.button("🗑️ Clear Database", type="secondary", help="Start clearing process"):
                    st.session_state.show_clear_confirmation = True
            else:
                st.info("💡 Database empty, nothing to clear")
        else:
            st.info("💡 Database not connected")

# Tabs
tab1, tab2, tab3 = st.tabs(["📤 Upload Protein Data", "🔍 Protein Sequence Search", "🛠️ Database Management"])

# Tab 1: Upload
with tab1:
    st.header("Upload Protein FASTA File")

    if st.session_state.upload_success:
        st.success(st.session_state.success_message)
        if st.session_state.upload_stats:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Uploaded This Time", st.session_state.upload_stats.get('inserted_count', 0),
                         delta=f"+{st.session_state.upload_stats.get('inserted_count', 0)}")
            with col2:
                st.metric("Total in DB", st.session_state.upload_stats.get('total_proteins', 0))
            with col3:
                st.metric("Success Rate", "100%")
        st.info("💡 **Next steps:**")
        st.markdown("- 🔍 Go to 'Protein Sequence Search' tab to test search")
        st.markdown("- 📤 Continue uploading more FASTA files")
        st.markdown("- 🛠️ Check 'Database Management' tab for stats")
        if st.button("✅ Got it, continue", type="secondary"):
            st.session_state.upload_success = False
            st.session_state.success_message = ""
            st.session_state.upload_stats = {}
            st.rerun()

    st.info("🤖 **Automated Handling:** System checks DB status and creates it if needed, then generates ESM2 embeddings.")
    st.markdown("Supports standard FASTA format. ESM2 embeddings will be generated and stored.")

    uploaded_file = st.file_uploader(
        "Select FASTA file",
        type=["fasta", "fa", "txt"],
        help="Supports .fasta, .fa, .txt"
    )

    if uploaded_file is not None:
        st.info(f"📁 Filename: {uploaded_file.name}")
        st.info(f"📊 File size: {uploaded_file.size} bytes")
        file_content = uploaded_file.read().decode("utf-8")
        with st.expander("🔍 Preview (first 500 chars)"):
            st.text(file_content[:500] + "..." if len(file_content) > 500 else file_content)
        process_button = st.button("🚀 Process & Upload", type="primary", use_container_width=True)
        if process_button:
            if "current_file_content" not in st.session_state:
                st.session_state.current_file_content = file_content
            if not st.session_state.get("processing_started", False):
                st.session_state.upload_success = False
                st.session_state.processing_started = True
            progress_bar = st.progress(0)
            status_text = st.empty()
            try:
                status_text.text("📋 Parsing FASTA file...")
                progress_bar.progress(20)
                protein_data = service.process_fasta_file(file_content)
                if protein_data:
                    status_text.text("✅ FASTA parsing complete")
                    progress_bar.progress(40)
                    st.success(f"✅ Parsed {len(protein_data)} protein sequences")
                    df_preview = pd.DataFrame([
                        {
                            "Protein ID": p["protein_id"],
                            "Length": p["length"],
                            "Description": p["description"][:50] + "..." if len(p["description"]) > 50 else p["description"]
                        } for p in protein_data[:5]
                    ])
                    st.dataframe(df_preview, use_container_width=True)
                    if len(protein_data) > 5:
                        st.info(f"📊 Showing first 5, total parsed {len(protein_data)}")
                    status_text.text("💾 Saving to database...")
                    progress_bar.progress(80)
                    inserted_count = service.insert_proteins(protein_data)
                    progress_bar.progress(100)
                    status_text.text("🎉 Processing complete!")
                    if inserted_count > 0:
                        refresh_stats()
                        updated_stats = get_real_time_stats()
                        st.session_state.upload_success = True
                        st.session_state.success_message = f"🎉 **Upload successful!** Inserted {inserted_count} protein records"
                        st.session_state.upload_stats = {
                            'inserted_count': inserted_count,
                            'total_proteins': updated_stats.get("total_proteins", 0)
                        }
                        st.session_state.processing_completed = True
                        st.session_state.processing_started = False
                        progress_bar.empty()
                        status_text.empty()
                        st.success(st.session_state.success_message)
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Uploaded", inserted_count, delta=f"+{inserted_count}")
                        with col2:
                            st.metric("DB Total", updated_stats.get("total_proteins", 0))
                        with col3:
                            st.metric("Success Rate", "100%")
                        st.info("💡 **Next steps:**")
                        st.markdown("- 🔍 Go to 'Protein Sequence Search'")
                        st.markdown("- 📤 Upload more FASTA files")
                        st.markdown("- 🛠️ Check 'Database Management'")
                        st.balloons()
                        st.rerun()
                    else:
                        progress_bar.empty()
                        status_text.empty()
                        st.error("❌ Upload failed - check format or DB connection")
                        st.session_state.processing_started = False
                else:
                    status_text.text("❌ Parsing failed")
                    progress_bar.progress(0)
                    st.error("❌ FASTA parsing failed, check format")
                    st.session_state.processing_started = False
            except Exception as e:
                status_text.text(f"❌ Processing error: {str(e)}")
                progress_bar.progress(0)
                st.error(f"❌ Error occurred: {str(e)}")
                st.session_state.processing_started = False

# Tab 2: Search
with tab2:
    st.header("Protein Sequence Similarity Search")
    service = get_protein_service()
    connection_status = service.check_database_connection()
    if connection_status["db_connected"]:
        stats = service.get_collection_stats()
        if stats.get("total_proteins", 0) == 0:
            st.warning("⚠️ No proteins in database. Upload data first.")
        else:
            st.success(f"🗄️ Database ready with {stats.get('total_proteins', 0)} records")
    else:
        st.info("💡 Database not connected yet. It will auto-connect on first search.")
    st.markdown("Enter a protein sequence to retrieve top-K similar sequences.")
    query_sequence = st.text_area(
        "Input query sequence",
        height=150,
        placeholder="Enter amino acid sequence\nExample: MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        help="Supports standard single-letter amino acid codes"
    )
    col1, col2 = st.columns([3, 1])
    with col1:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    with col2:
        top_k = st.selectbox(
            "Result count (K)",
            options=list(range(1, 21)),
            index=4,
            help="Select number of similar sequences to return"
        )
    if search_button and query_sequence.strip():
        with st.spinner(f"Searching top {top_k} similar proteins..."):
            results = service.search_similar_proteins(query_sequence.strip(), top_k)
            if results:
                st.success(f"🎯 Found {len(results)} similar proteins")
                for i, result in enumerate(results, 1):
                    with st.expander(f"🏆 Rank #{i} - {result['protein_id']} (Similarity: {result['similarity_score']:.4f})"):
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.markdown("**Basic Info:**")
                            st.write(f"🆔 **Protein ID:** {result['protein_id']}")
                            st.write(f"📏 **Length:** {result['length']} aa")
                            st.write(f"🎯 **Similarity:** {result['similarity_score']:.6f}")
                        with col2:
                            st.markdown("**Sequence:**")
                            if result.get('description'):
                                st.write(f"📝 **Description:** {result['description']}")
                            sequence = result.get('sequence', '')
                            if len(sequence) > 100:
                                st.text_area(
                                    f"🧬 Sequence (first 100 aa):",
                                    sequence[:100] + "...",
                                    height=100,
                                    disabled=True,
                                    key=f"sequence_short_{i}"
                                )
                                st.caption(f"Full length: {len(sequence)} aa")
                            else:
                                st.text_area(
                                    "🧬 Full sequence:",
                                    sequence,
                                    height=100,
                                    disabled=True,
                                    key=f"sequence_full_{i}"
                                )
            else:
                st.warning("❌ No similar proteins found. Try another sequence.")
    elif search_button and not query_sequence.strip():
        st.error("❌ Please enter a valid sequence")

# Tab 3: Database Management
with tab3:
    st.header("Database Management & Statistics")
    service = get_protein_service()
    connection_status = service.check_database_connection()
    if connection_status["db_connected"]:
        stats = service.get_collection_stats()
        st.success("🗄️ Database connected")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Protein Count", stats.get("total_proteins", 0))
        with col2:
            st.metric("Collection", value="", help=stats.get("collection_name", "N/A"))
            st.write(f"📁 {stats.get('collection_name', 'N/A')}")
        with col3:
            st.metric("Load State", value="", help="Collection loaded in memory")
            st.write("✅ Loaded" if stats.get("is_loaded", False) else "❌ Not Loaded")
        with col4:
            st.metric("DB File", value="", help="Milvus Lite database file path")
            st.write("📄 Local file")
        with st.expander("📊 Detailed Statistics", expanded=True):
            info_data = {
                "Item": ["Collection", "Protein Count", "Index Type", "Metric", "Vector Dim", "Load State"],
                "Value": [
                    str(stats.get("collection_name", "N/A")),
                    str(stats.get("total_proteins", 0)),
                    "AUTOINDEX (Milvus Lite)",
                    "L2 (Euclidean)",
                    "320 (ESM2-t6-8M)",
                    "Loaded" if stats.get("is_loaded", False) else "Not Loaded"
                ]
            }
            df_info = pd.DataFrame(info_data)
            st.dataframe(
                df_info,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Item": st.column_config.TextColumn("Item", width="medium"),
                    "Value": st.column_config.TextColumn("Value", width="large")
                }
            )
        st.markdown("---")
        st.subheader("🔧 Database Actions")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh Stats", use_container_width=True):
                refresh_stats()
                st.rerun()
        with col2:
            if stats.get("total_proteins", 0) > 0:
                if st.button("🗑️ Clear Database", type="secondary", use_container_width=True):
                    st.session_state.show_clear_confirmation = True
        if st.session_state.get('show_clear_confirmation', False):
            st.markdown("---")
            st.error("⚠️ **Danger Operation Confirmation**")
            st.warning("You are about to clear the database. This cannot be undone.")
            if 'verification_code' not in st.session_state:
                st.session_state.verification_code = ''.join(random.choices(string.digits, k=6))
            st.info(f"🔢 Enter verification code: **{st.session_state.verification_code}**")
            user_code = st.text_input("Verification Code", max_chars=6, placeholder="Enter 6 digits")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.show_clear_confirmation = False
                    if 'verification_code' in st.session_state:
                        del st.session_state.verification_code
                    st.rerun()
            with col2:
                if st.button("🗑️ Confirm Clear", type="secondary", use_container_width=True, disabled=user_code != st.session_state.verification_code):
                    if user_code == st.session_state.verification_code:
                        try:
                            success = service.clear_database()
                            if success:
                                st.success("✅ Database cleared")
                                st.session_state.show_clear_confirmation = False
                                if 'verification_code' in st.session_state:
                                    del st.session_state.verification_code
                                refresh_stats()
                                st.rerun()
                            else:
                                st.error("❌ Clear failed")
                        except Exception as e:
                            st.error(f"❌ Error during clear: {str(e)}")
                    else:
                        st.error("❌ Incorrect code")
    else:
        st.error("❌ Database connection failed")
        st.info("💡 Check system status or restart application")

# Global clear confirmation dialog
if st.session_state.get("show_clear_confirmation", False):
    if "verification_code" not in st.session_state:
        st.session_state.verification_code = ''.join(random.choices(string.digits, k=6))
    st.markdown("---")
    st.markdown("## ⚠️ Confirm Database Clear")
    with st.container():
        st.error("### ⚠️ Danger Operation Confirmation")
        st.markdown("You are about to remove all protein data. This action is **irreversible**.")
        stats = service.get_collection_stats()
        total_proteins = stats.get("total_proteins", 0)
        st.warning(f"📊 Will delete **{total_proteins}** protein records")
        st.markdown("---")
        st.markdown("### 🔐 Security Verification")
        st.info(f"Enter code: **{st.session_state.verification_code}**")
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            user_code = st.text_input(
                "Verification Code",
                placeholder="Enter 6-digit code",
                max_chars=6,
                key="verification_input"
            )
        with col2:
            confirm_button = st.button(
                "🗑️ Confirm Clear",
                type="primary",
                disabled=(user_code != st.session_state.verification_code),
                use_container_width=True
            )
        with col3:
            cancel_button = st.button(
                "❌ Cancel",
                type="secondary",
                use_container_width=True
            )
        if confirm_button and user_code == st.session_state.verification_code:
            with st.spinner("Clearing database..."):
                success = service.clear_database()
            if success:
                st.success("✅ Database cleared successfully!")
                st.balloons()
                st.session_state.show_clear_confirmation = False
                if "verification_code" in st.session_state:
                    del st.session_state.verification_code
                if "verification_input" in st.session_state:
                    del st.session_state.verification_input
                st.info("Refreshing page...")
                st.rerun()
            else:
                st.error("❌ Clear failed, check logs")
        elif cancel_button:
            st.session_state.show_clear_confirmation = False
            if "verification_code" in st.session_state:
                del st.session_state.verification_code
            if "verification_input" in st.session_state:
                del st.session_state.verification_input
            st.rerun()
        if user_code and user_code != st.session_state.verification_code:
            st.error("❌ Verification code mismatch")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🧬 Protein RAG Retrieval System | Powered by ESM2 & Milvus | Efficient protein similarity search</p>
    </div>
    """,
    unsafe_allow_html=True
)
