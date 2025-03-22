import streamlit as st
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer


indexName = "all_products"

try:
    es = Elasticsearch("https://localhost:9200", basic_auth=('elastic', 'mvsFaXzEkUiSOhanK2lS'), verify_certs=False)
except ConnectionError as e:
    st.error(f"Connection Error: {e}")
    
if es.ping():
    st.success("Successfully connected to Elasticsearch!")
else:
    st.error("Oops! Cannot connect to Elasticsearch!")

def search(input_keyword):
    model = SentenceTransformer('all-mpnet-base-v2')
    vector_of_input_keyword = model.encode(input_keyword)

    query = {
        "field": "DescriptionVector",
        "query_vector": vector_of_input_keyword,
        "k": 5,
        "num_candidates": 500
    }
    res = es.knn_search(index="all_products"
                        , knn=query 
                        , source=["ProductName","Description"]
                        )
    results = res["hits"]["hits"]
    return results

def main():
    # Use custom CSS styling
    st.markdown("""
        <style>
            .main {
                background-color: #f7f7f7;
                padding: 20px;
                color:black;
            }
            .stButton>button {
                background-color: #3399ff;
                color: white;
                font-size: 16px;
                padding: 12px;
                border-radius: 8px;
                width:8rem;
                height:2rem;
            }
            .stButton>button:hover {
                background-color: #0080ff;
                color: white;
                border-color: #0080ff;
            }
            .search-result-header {
                font-size: 24px;
                color: #333;
                font-weight: bold;
            }
            .product-description {
                font-size: 16px;
                color: #555;
                margin-top: 5px;
            }
        </style>
    """, unsafe_allow_html=True)


    st.markdown("<div>", unsafe_allow_html=True)


    st.title("Myntra Fashion")
    st.subheader("Explore Fashion with SBERT & Elasticsearch")
    search_query = st.text_input("Enter your search query:")

    if st.button("Search"):
        if search_query:

                results = search(search_query)
                if results:
                    st.subheader("Search Results")
                    for result in results:
                        with st.container():
                            if '_source' in result:
                                try:
                                    st.markdown(f"<div>", unsafe_allow_html=True)
                                    st.markdown(f"<div class='search-result-header'>{result['_source']['ProductName']}</div>", unsafe_allow_html=True)
                                except Exception as e:
                                    print(e)
                                
                                try:
                                    st.markdown(f"<div class='product-description'>{result['_source']['Description']}</div>", unsafe_allow_html=True)
                                except Exception as e:
                                    print(e)
                            st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.warning("No results found.")

    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
