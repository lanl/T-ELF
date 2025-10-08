import streamlit as st
from streamlit_tree_select import tree_select
import pandas as pd
import os
import numpy as np
from pages.helpers.utils import (find_files_by_extensions, 
                                 prune_tree, filter_documents,
                                 extract_unique_values)
from pages.helpers.displays import (display_html,
                                    display_dataframe, display_wordcloud,
                                    open_explorer_button, display_images_in_tabs,
                                    display_html_selector, display_csv_in_tab,
                                    display_document_years, display_attributes,
                                    display_denovo_words, display_author_chart,
                                    display_interactive_coauthorship_plot,
                                    display_document_affiliations,
                                    display_coauthorship_heatmap)


st.set_page_config(layout="wide")
if ("project_loaded" in st.session_state and st.session_state.project_loaded == False) or "data" not in st.session_state:
    st.warning("⚠️ Data has not been loaded yet. Please load the data before accessing this page.")
    st.stop()

if st.session_state.view_type != "Document Analysis":
    st.warning("⚠️ Document Analysis view mode is not selected.")
    st.stop()

st.header(st.session_state.selected_project)
with st.sidebar:
    st.header("Token Search")

    top_n = st.selectbox("Top n:", ["All"] + list(range(1, 101)), index=0)
    keywords_search_type = st.selectbox("Word Search Type:", ["Keywords", "Denovo"], index=0)
    if keywords_search_type == "Keywords":
        token_search_field = "all_words"
    else:
        token_search_field = "all_words_denovo"
    
    # Gather all unique words
    words = list(
        set(
            w
            for wlist in st.session_state.documents_data[token_search_field].values()
            for w in wlist
        )
    )
    words.sort()

    selected_words = st.multiselect("Select words to search:", words)
    negative_words = st.multiselect("Select negative words:", words)
    token_search_type = st.radio("Search type:", ["and", "or"])

    # ---------------------------------------------------
    # Attribute Search
    st.header("Attribute Search")

    search_categories = {
        "Countries": {
            "options": extract_unique_values("all_countries", st.session_state.documents_data),
            "key": "country"
        },
        "Authors": {
            "options": extract_unique_values("all_authors", st.session_state.documents_data),
            "key": "author"
        },
        "Affiliations": {
            "options": extract_unique_values("all_affiliations", st.session_state.documents_data),
            "key": "affiliation"
        },
        f"Attribute {st.session_state.attribute_column}": {
            "options": extract_unique_values("all_attributes", st.session_state.documents_data),
            "key": "attributes"
        }
    }

    # Build UI for attribute selections
    attr_selections = {}
    for label, info in search_categories.items():
        with st.expander(f"Search by {label}"):
            attr_key = info["key"]
            attr_selections[f"selected_{attr_key}"] = st.multiselect(
                f"Select {label.lower()}:", info["options"]
            )
            attr_selections[f"{attr_key}_search_type"] = st.radio(
                "Search type:", ["and", "or"], key=f"{attr_key}_radio"
            )
    # ---------------------------------------------------
    # Other Settings
    st.header("Other Settings")
    filter_df_setting = st.checkbox("Filter Dataframe", value=True, key="filter_dataframe_settings")


# ---------------------------------------------------------------------
# Determine if the user selected anything
any_token_selected = bool(selected_words)
any_negative_token_selected = bool(negative_words)
any_attr_selected = any(
    attr_selections.get(f"selected_{cfg['key']}", [])
    for cfg in search_categories.values()
)

# If nothing is selected, reset the data
if not any_token_selected and not any_attr_selected and not negative_words:
    st.session_state.data = st.session_state.data_original.copy()
    st.session_state.selected_indices = None
else:
    # Filter documents once based on both token and attribute selections
    selected_labels, selected_indices = filter_documents(
        data_map=st.session_state.data_map,
        top_n=top_n,
        selected_words=selected_words,
        token_search_type=token_search_type,
        attr_selections=attr_selections,
        search_categories=search_categories,
        keywords_search_type=keywords_search_type,
        use_index_map=filter_df_setting,
        negative_words=negative_words,
    )
    if selected_labels:
        st.session_state.data = [prune_tree(st.session_state.data_original[0], selected_labels, st.session_state.root_name)]
        st.session_state.selected_indices = selected_indices
    else:
        # No matches => clear data or handle how you prefer
        st.session_state.data = [{"children":[]}]
        st.session_state.selected_indices = None
    #st.write(selected_indices)
    
selected_nodes = tree_select(
    st.session_state.data[0]["children"], 
    expand_on_click=False,
    show_expand_all=True,
    check_model="all",
    no_cascade=True
)

tabs = st.tabs(["Words", "Documents", "Peacock", "Attributes", "Authors", "Affiliations"])
st.session_state.html_figure = None  # Reset HTML viewer

### WORDS TAB
with tabs[0]:
    for ii in selected_nodes["checked"]:
        directory = st.session_state.data_map[ii]["path"]
        with st.expander(st.session_state.data_map[ii]["label"]):
            open_explorer_button(directory, key=f"button_tab0_{ii}")
            
            word_tab_columns = st.columns(3)
            # Display unigrams
            with word_tab_columns[0]:
                display_dataframe(directory, suffix="_bow_unigrams.csv", title="Unigrams", columns=["word", "tf", "df"], ends=True)
            
            # Display bigrams
            with word_tab_columns[1]:
                display_dataframe(directory, suffix="_bow_bigrams.csv", title="Bigrams", columns=["word", "tf", "df"], ends=True)
            
            # Display word cloud
            with word_tab_columns[2]:
                display_wordcloud(directory)
            
            if st.session_state.data_map[ii]["all_text"] is not None and len(st.session_state.data_map[ii]["all_text"]) > 0:
                display_denovo_words(st.session_state.data_map[ii]["all_text"], key=f"slider_words{[ii]}")

### DOCUMENTS TAB
with tabs[1]:
    for ii in selected_nodes["checked"]:
        directory = st.session_state.data_map[ii]["path"]
        label = st.session_state.data_map[ii]["label"]
        index_filter = None if st.session_state.selected_indices is None else st.session_state.selected_indices[label]
        with st.expander(label):
            open_explorer_button(directory, key=f"button_tab1_{ii}")
            display_dataframe(directory, suffix=st.session_state.document_file_name_suffix, 
                              title="Top Documents", columns=None, ends=False,
                              index_filter=index_filter)
            display_document_years(directory, suffix=st.session_state.document_file_name_suffix, 
                                   title="Top Documents Trends", ends=False, column="year", 
                                   key=f"tab1{ii}_document_years", index_filter=index_filter)

### PEACOCK TAB
with tabs[2]:
    
    for ii in selected_nodes["checked"]:
        directory = st.session_state.data_map[ii]["path"]
        peacock_dir = os.path.join(directory, "peacock")
        print( peacock_dir )
        files = find_files_by_extensions(peacock_dir, extensions=("html", "png"))
        with st.expander(st.session_state.data_map[ii]["label"]):
            open_explorer_button(peacock_dir, key=f"button_tab2_{ii}")
            all_tabs = [os.path.basename(f) for f in files["png"]] + ["Interactive Plots", "Affiliation", "Author"]

            if all_tabs:
                tab_objects = st.tabs(all_tabs)

                # Display images in tabs
                display_images_in_tabs(tab_objects, files["png"])

                # Display interactive plots selector
                display_html_selector(tab_objects[-3], files["html"], key=f"select_html_plot{ii}")

                # Display affiliation and author data
                display_csv_in_tab(tab_objects[-2], peacock_dir, "_affiliation.csv", "Affiliations Peacock file is not found.", ends=True)
                display_csv_in_tab(tab_objects[-1], peacock_dir, "_author.csv", "Authors Peacock file is not found.", ends=True)

### ATTRIBUTES TAB
with tabs[3]:
    for ii in selected_nodes["checked"]:
        directory = st.session_state.data_map[ii]["path"]
        with st.expander(st.session_state.data_map[ii]["label"]):
            open_explorer_button(directory, key=f"button_tab3_{ii}")
            if len(st.session_state.data_map[ii]["attributes"]) > 0:
                st.header(st.session_state.attribute_column)
                display_attributes(st.session_state.data_map[ii]["attributes"], key=f"tab3_attributes_{ii}")
            else:
                st.warning(f"⚠️ Attributes from column {st.session_state.attribute_column} not found!")

### AUTHORS TAB
with tabs[4]:
    for ii in selected_nodes["checked"]:
        directory = st.session_state.data_map[ii]["path"]
        with st.expander(st.session_state.data_map[ii]["label"]):
            open_explorer_button(directory, key=f"button_tab4_{ii}")
            display_author_chart(directory=directory, 
                                     suffix=st.session_state.document_file_name_suffix, 
                                     title="Author Distribution Chart", 
                                     ends=False, columns=["authors", "author_ids"],
                                     key=f"tab4_author_{ii}")
            display_interactive_coauthorship_plot(directory=directory, 
                                     suffix=st.session_state.document_file_name_suffix, 
                                     title="Co-authorship Network", 
                                     ends=False, columns=["authors", "author_ids"],
                                     key=f"tab4_coauthor_{ii}")
            display_coauthorship_heatmap(directory=directory, 
                                     suffix=st.session_state.document_file_name_suffix, 
                                     title="Co-authorship Heatmap", 
                                     ends=False, columns=["authors", "author_ids"],
                                     key=f"tab4_coauthor_heatmap_{ii}")
            

### AFFILIATIONS TAB
with tabs[5]:
    for ii in selected_nodes["checked"]:
        directory = st.session_state.data_map[ii]["path"]
        with st.expander(st.session_state.data_map[ii]["label"]):
            open_explorer_button(directory, key=f"button_tab5_{ii}")
            display_document_affiliations(directory=directory, 
                                     suffix=st.session_state.document_file_name_suffix, 
                                     title="Affiliation Plots", 
                                     ends=False, column="affiliations",
                                     key=f"tab5_affiliations_{ii}")

### DISPLAY HTML COMPONENT FULL WIDTH (NO JAVASCRIPT)
if st.session_state.html_figure:
    st.divider()
    display_html(path=st.session_state.html_figure, height=800)
