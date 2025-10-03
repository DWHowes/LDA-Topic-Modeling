import pandas as pd
import numpy as np

import streamlit as st

from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.metrics.pairwise import cosine_similarity

#visualization
import pyLDAvis
import pyLDAvis.lda_model as pyLDA

from wordcloud import WordCloud

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from plotly.figure_factory import create_dendrogram

from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage

class docModel:
    def __init__(self, fname:str) -> None:
        self.ngram_vec = None
        self.ngram_matrix = None
        self.model = None
        self.span_list = None
        self.dominant_topics = None
        self.doc_count = 0

        self.span_list = self.__get_paragraphs(fname)

        self.vis_method_map = {
            "topic map": self.__topic_map,
            "topic similarity": self.__topic_similarity,
            "topic barchart": self.__topic_barchart,
            "topic clouds": self.__word_clouds,
            "topic sunburst": self.__sunburst,
            "topic treemap": self.__treemap,
            "document topics": self.__document_topics,
            "documents": self.__documents,
            "3D document topics": self.__3d_topic_map,
            "cluster map": self.__cluster_map  
        }

## PRIVATE METHODS

    def __topic_map(self):
        # Visualize the model on StreamLit
        vis = pyLDA.prepare(self.model, 
                            self.ngram_matrix, 
                            self.ngram_vec, 
                            mds='mmds', 
                            R=st.session_state.p_terms)
        st.session_state.p_vis_html = pyLDAvis.prepared_data_to_html(vis, template_type='general')        

    def __topic_similarity(self)->go.Figure:
        st.session_state.p_fig_plotly = True

        topic_word_distribution = self.model.components_

        cosine_similarities = cosine_similarity(topic_word_distribution)

        data = go.Heatmap(z=cosine_similarities, colorscale='viridis_r')
        layout = go.Layout(width=950, height=950, title="Topic Similarity", xaxis=dict(title="Topic"), yaxis=dict(title="Topic"))

        return go.Figure(dict(data=[data], layout=layout))   

    def __document_topics(self)->go.Figure:
        st.session_state.p_fig_plotly = True

        counts = self.dominant_topics.Dominant_Topic.value_counts()

        chart_df = pd.DataFrame(columns = ["topic", "value", "keywords"])

        for i in range(len(counts)):
            topic = i
            value = counts[i]
            keywords = self.dominant_topics.iloc[i+1, self.dominant_topics.columns.get_loc('Keywords')]
            chart_df.loc[len(chart_df)] = [topic, value, keywords] 

        fig = px.bar(chart_df, 
                     x='topic', 
                     y='value', 
                     title='Documents Grouped by Dominant Topic', 
                     hover_data=['keywords'], 
                     labels={'value': 'No. of Documents', 'topic': 'Topic Number'}, 
                     width=750, 
                     height=500)
        fig.update_traces(width=.5)
        fig.update_layout(bargap=.1)
        fig.update_xaxes(type='category')

        return fig
    
    def __topic_barchart(self)->go.Figure:
        st.session_state.p_fig_plotly = True

        # Topic-Keyword Matrix
        keywords = np.array(self.ngram_vec.get_feature_names_out())

        # Get the top 5 keywords for each topic
        topic_keywords = []
        for topic_weights in self.model.components_:
            top_keyword_locs = (-topic_weights).argsort()[:5]
            topic_keywords.append((keywords.take(top_keyword_locs), topic_weights.take(top_keyword_locs)))

        # Topic keywords and weights Dataframe
        df_topic_keywords = pd.DataFrame(topic_keywords)
        df_topic_keywords.columns = ['Keywords', 'Weights']

        # Dynamically allocate the number of rows, columns are fixed at 3
        num_plots = self.model.n_components
        n_cols = 3
        n_rows = int((self.model.n_components + n_cols - 1)/n_cols)

        # Dynamically create the subplot title list
        subplot_titles_list = []
        for i in range(num_plots):
            subplot_titles_list.append("Topic "+str(i)) 

        # Create the figure with the required number of subplots
        fig = make_subplots(rows=n_rows, 
                            cols=n_cols, 
                            vertical_spacing=0.08, 
                            subplot_titles=subplot_titles_list)

        # Dynamically allocate subplots, iterating through the rows
        for i, row_lists in df_topic_keywords.iterrows():
            col = ((i+n_cols)%n_cols)+1
            row = (i+n_cols)//n_cols
            fig.add_trace(go.Bar(x=row_lists.Weights, 
                                 y=row_lists.Keywords, 
                                 orientation='h'), 
                                 row=row, 
                                 col=col)

        fig.update_layout(uniformtext_minsize=4, uniformtext_mode='hide', showlegend=False, height=1200)    

        return fig    
    
    def __documents(self)->go.Figure:
        st.session_state.p_fig_plotly = True

        doc_topic_dist = self.model.fit_transform(self.ngram_matrix)
        topic_num = doc_topic_dist.argmax(axis=1)

        # tSNE Dimension Reduction
        tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
        tsne_lda = tsne_model.fit_transform(doc_topic_dist)
        tsne_df = pd.DataFrame(data=tsne_lda, columns=['x', 'y'])
        tsne_df['topic'] = topic_num
        tsne_df['topic'] = tsne_df['topic'].astype(str)

        fig = px.scatter(tsne_df, 
                         x='x', 
                         y='y', 
                         color='topic', 
                         hover_data=['topic'], 
                         height=800, 
                         width=1000, 
                         labels={"x": "Component 1", "y": "Component 2"})        

        fig.update_layout(title=dict(text="LDA Document Clusters"))

        return fig
    
    # Helper function for __word_clouds()
    # For a given topic, return the top 10 words and their weights
    def __get_word_weights(self, feature_names, topic)->dict:
        # Get words and their importance for the current topic
        top_words_indices = topic.argsort()[:-10:-1]  # Top 10 words
        top_words = [(feature_names[i], topic[i]) for i in top_words_indices]

        # Create a dictionary
        word_weights = {word: weight for word, weight in top_words}
        
        return word_weights

    def __word_clouds(self)->Figure:
        st.session_state.p_fig_plotly = False

        feature_names = self.ngram_vec.get_feature_names_out()
        topic_word_distributions = self.model.components_

        cloud = WordCloud(background_color='white', width=2500, height=1800)

        # dynamically allocate subplots
        n_cols = 4
        n_rows = int((self.model.n_components + n_cols - 1)/n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10,10), sharex=True, sharey=True)

        for i, ax in enumerate(axes.flatten()):
            if i < self.model.n_components:
                fig.add_subplot(ax)
                # Get words and their importance for the current topic
                word_weights = self.__get_word_weights(feature_names, topic_word_distributions[i])
                # Generate and display word cloud
                cloud.generate_from_frequencies(word_weights)
                plt.gca().imshow(cloud, interpolation="bilinear")
                plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=12))
                plt.gca().axis('off')
                rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes, edgecolor='gray', facecolor='none', lw=1.5)
                # Add the rectangle to the axes
                ax.add_patch(rect)
            else:
                # Remove unused subplots
                fig.delaxes(ax)

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout()

        return fig
    
    def __sunburst(self)->go.Figure:
        st.session_state.p_fig_plotly = True

        topic_word_imp = self.__get_vis_data()
        topic_word_imp['hole'] = 'Topics'

        fig = px.sunburst(topic_word_imp, 
                        path=['hole', 'topic', 'word'], 
                        values='importance', 
                        color='importance', 
                        color_continuous_scale='viridis', 
                        labels={'importance': 'Importance', 'name': 'Name'},
                        hover_data=['name']
                        )        
        
        fig.update_layout(width=900, 
                          height=900, 
                          margin = dict(t=0, l=0, r=0, b=0), 
                          coloraxis_colorbar = dict(len=0.8, thickness=20)
                          )
        
        return fig
    
    def __treemap(self)->go.Figure:
        st.session_state.p_fig_plotly = True

        treemap_data = self.__get_vis_data()

        fig = px.treemap(treemap_data, path=['topic', 'word'],
                        values='importance',
                        color='importance', 
                        hover_data=['name'],
                        color_continuous_scale='viridis',
                        color_continuous_midpoint=np.average(treemap_data['importance'])
                        )
        
        fig.update_layout(margin = dict(t=0, l=0, r=0, b=0),
                          coloraxis_colorbar = dict(len=0.8, thickness=20)                          
                          )        

        return fig
    
    def __3d_topic_map(self)->go.Figure:
        st.session_state.p_fig_plotly = True

        doc_topic_dist = self.model.fit_transform(self.ngram_matrix)

        topic_num = doc_topic_dist.argmax(axis=1)

        # t-SNE Dimension Reduction
        tsne_model = TSNE(n_components=3, verbose=1, random_state=0, angle=.99, init='pca')
        tsne_lda = tsne_model.fit_transform(doc_topic_dist)
        tsne_df = pd.DataFrame(data=tsne_lda, columns=['x', 'y', 'z'])
        tsne_df['topic'] = topic_num
        tsne_df['topic'] = tsne_df['topic'].astype(str)

        fig = px.scatter_3d(tsne_df, 
                            x='x', 
                            y='y', 
                            z='z', 
                            color='topic', 
                            hover_data=['topic'], 
                            height=1000, 
                            labels={"x": "Component 1", "y": "Component 2", "z": "Component 3"}
                            )  
        
        fig.update_layout(title=dict(text="LDA 3-D Document Clusters"))

        return fig      
    
    def __cluster_map(self)->go.Figure:
        st.session_state.p_fig_plotly = True
        Index = [str(i) for i in range(self.model.n_components)]

        topic_word_distributions = self.model.components_

        # Use cosine distance as the distance function
        cosine_distance = lambda x: 1 - cosine_similarity(x)
        # Use Ward as the linkage function. Ward minimizes variance within clusters
        linkage_function = lambda x: linkage(x, "ward", optimal_ordering=True)

        df = pd.DataFrame(topic_word_distributions, index=Index)

        fig = create_dendrogram(df, distfun=cosine_distance, linkagefun=linkage_function)
        fig.update_layout(width=1000, 
                          height=800, 
                          xaxis_title='Topics', 
                          yaxis_title='Cosine Distance', 
                          title='Cluster Map of LDA Topics')

        return fig      

    def __get_paragraphs(self, fname:str):
        df = pd.read_json(fname)
        self.doc_count = len(df)
        return df.get("paragraphs")

    # Helper method to create the Pandas dataframe used by the sunburst and treemap methods
    def __get_vis_data(self)->pd.DataFrame:
        feature_names = self.ngram_vec.get_feature_names_out()
        topic_word_distributions = self.model.components_
        num_topics = self.model.n_components
        
        topic_word_freq = pd.DataFrame(columns=['topic', 'word', 'importance', 'name'])

        for topic_idx, topic in enumerate(topic_word_distributions):
            name = "Topic " + str(topic_idx)
            # Get words and their importance for the current topic
            top_words_indices = topic.argsort()[:-num_topics:-1] 
            top_words = [(feature_names[i], topic[i]) for i in top_words_indices]

            for word, importance in top_words:
                topic_word_freq.loc[len(topic_word_freq)] = [topic_idx, word, importance, name] 
                
        return topic_word_freq    
    
    # Helper method to return a pandas dataframe of the dominant topic in each document
    def __get_dominant_topic(self)->pd.DataFrame:
        # Output dataframe
        dominant_topics = pd.DataFrame(columns=['Document', 'Dominant_Topic', 'Weight', 'Keywords'])
        
        # Create Document - Topic Matrix
        lda_output = self.model.transform(self.ngram_matrix, normalize=False)

        # index of documents
        docnames = [str(i) for i in range(len(self.span_list))]
        # Create the intermediate dataframe
        df_document_topic = pd.DataFrame(np.round(lda_output, 2))
        # Get dominant topic for each document
        dominant_topic = np.argmax(df_document_topic.values, axis=1)
        # Find the largest value in each row
        weight_list = df_document_topic.max(axis=1).tolist()
        # Topic keywords
        keywords = np.array(self.ngram_vec.get_feature_names_out())
        topic_keywords = []
        for topic_weights in self.model.components_:
            top_keyword_locs = (-topic_weights).argsort()[:5]
            topic_keywords.append(', '.join((keywords.take(top_keyword_locs))))     

        # Assign columns to output dataframe
        dominant_topics['Document'] = docnames
        dominant_topics['Dominant_Topic'] = dominant_topic
        dominant_topics['Weight'] = weight_list
        for i, keyidx in enumerate(dominant_topic):
            dominant_topics.loc[i, 'Keywords'] = topic_keywords[keyidx]

        return dominant_topics        
    
# PUBLIC METHODS

    def lda_model(self, num_topics=10, update=1, chunks=100, passes=10):
        # Create a document-term matrix - BoW format. Composed of bigrams and trigrams
        self.ngram_vec = CountVectorizer(stop_words='english', ngram_range=(2,3))
        self.ngram_matrix = self.ngram_vec.fit_transform(self.span_list)
        #Create and fit the LDA model
        model = LDA(n_components=num_topics, 
                    max_iter=passes, 
                    evaluate_every=update, 
                    batch_size=chunks, 
                    random_state=42)
        self.model = model.fit(self.ngram_matrix)

        self.dominant_topics = self.__get_dominant_topic()

        return self.model
    
    def vis_data(self, key:str):
        return self.vis_method_map[key]()
