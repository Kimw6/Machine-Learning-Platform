import streamlit as st

st.subheader('Unsupervised Machine Learning')

import networkx as nx
import matplotlib.pyplot as plt

def plot_graph():
    # Create a sample graph
    G = nx.complete_graph(5)

    # Draw the graph
    plt.figure(figsize=(8, 6))
    nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_color='gray')
    plt.title('Sample Graph')
    return plt

def main():
    st.title("Graph Visualization with Streamlit")
    st.write("This is a simple graph visualization:")

    # Plot the graph
    st.pyplot(plot_graph())

if __name__ == '__main__':
    main()
