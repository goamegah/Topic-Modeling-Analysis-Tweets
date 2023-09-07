from dash import Dash, html, dcc, callback, Output, Input, CeleryManager, dash_table
import plotly.express as px
import pandas as pd
from tweemo.controleur import Controller
import dash_bootstrap_components as dbc
from io import BytesIO
import base64

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.LUX]
)
tb_name = "tweets_cluster_tb"

controller = Controller("tweets_app")
models_d = controller.get_modeles_name()
models_name = list(models_d.keys())
reduc_method_d = controller.get_reduc_method()
method_reduc_name = list(reduc_method_d.keys())

app.layout = html.Div(id="root_container", children=[  # All page container
    html.H1(children='Tweets App', style={'textAlign': 'center'}),
    html.Div([  # Option Dataset
        html.H2(children="Selectionner le nombre de lignes du dataset"),
        html.P(id='placeholder-load-df', style={"display": "none"}),
        dcc.Slider(
            min=1000, max=4500, step=101, value=1001,
            id='slider-selection-nrows',
            marks=
            {
                i: {"label": f"{i} rows", 'style': {'font-size': '14px'}}
                for i in range(1001, 15001) if i % 500 == 0
            }
        )
    ]),

    html.Div([  # Descriptive Window
        html.H2(children='Onglet Descriptif'),
        html.Div(id="table"),
        html.Div([  # histogram for tweets count
            dcc.Graph(id='hist-plot-tweets')
        ], style={"margin-bottom": "21px"}),
        html.Div([  # barchart for tweets
            dcc.Slider(1, 21, 1,
                       value=5,
                       id='slider-selection-ntokens',
                       marks={
                           i: {"label": f"{i} words", 'style': {'font-size': '14px'}}
                           for i in range(1, 22) if i % 2 == 0
                       }

                       ),
            dcc.Graph(id='bar-plot-tweets')

        ])

    ], style={"margin-bottom": "25px"}),

    html.H2(children='Onglet pour la Clusterisation'),
    html.Div([  # Model Window
        html.Div([  # Model Selection + Dropdown model + number topics
            html.Div([  # Model Selection + dropdown selection model
                html.Div([
                    html.Label("Model Selection")
                ], style={"margin-bottom": "7px"}),
                dcc.Dropdown(models_name, models_name[0],
                             id='dropdown-selection-models',
                             style={"width": "50%"},
                             clearable=False
                             )
            ], style={"margin-bottom": "15px"}),
            html.Div([  # Number Topics
                dcc.Slider(
                    id='slider-selection-number-topics',
                    min=1,
                    max=8,
                    value=3,
                    marks={
                        1: {'label': '1 Topic', 'style': {'font-size': '14px'}},
                        3: {'label': '3 Topic', 'style': {'font-size': '14px'}},
                        5: {'label': '5 Topic', 'style': {'font-size': '14px'}},
                        7: {'label': '7 Topic', 'style': {'font-size': '14px'}},
                    },
                    step=1,

                )
            ], style={"margin-bottom": "15px"}),
            html.P(id='placeholder-load-model', style={"display": "none"})
        ], style={"margin-bottom": "55px"}),
        html.Div([  # Histogram distribution by topics
            dcc.Graph(id="hist-plot-tweets-by-topics")
        ], style={"margin-bottom": "35px"}),
        html.Div([  # bar chart by topics

            dcc.Slider(1, 21, 1,
                       value=5,
                       id='slider-selection-ntokens-by-topics',
                       marks={
                           i: {"label": f"{i} words", 'style': {'font-size': '14px'}}
                           for i in range(1, 22) if i % 2 == 0
                       }
                       ),
            dcc.Graph(id='bar-plot-tweets-by-topics'),
            dcc.Slider(0, 8, 1,
                       value=2,
                       id='slider-selection-topic',
                       marks={
                           0: {'label': 'Topic 0', 'style': {'font-size': '14px'}},
                           3: {'label': 'Topic 3', 'style': {'font-size': '14px'}},
                           5: {'label': 'Topic 5', 'style': {'font-size': '14px'}},
                           7: {'label': 'Topic 7', 'style': {'font-size': '14px'}},
                       }
                       )

        ], style={"margin-bottom": "15px"}),

        # Word Cloud section
        html.Div([html.Img(id="image_wc_by_topics",
                           style={"width": "95%",
                                  "margin-left": "45px",
                                  "margin-right": "15px"}),], style={"margin-top": "15px"}),
        html.Div([  # tweets embedding section
            html.Div([  # drop down for reduction method container
                dcc.Dropdown(method_reduc_name, method_reduc_name[0],
                             id='dropdown-selection-reduc-method',
                             style={"width": "50%"},
                             clearable=False
                             )

            ]),
            dcc.Graph(id="scatter-plot-tweets-by-topics")
        ], style={"margin-top": "35px"}),
        html.Div(id="metrics")  # metrics of clustering model
    ]),

], style={"margin-left": "15px"})


# Option Dataset

@callback(
    Output("placeholder-load-df", "children"),
    Input("slider-selection-nrows", "value")
)
def load_df(number_rows: int):
    rows_class = int(number_rows / 3)
    controller.df = controller.get_df(
        tb_name=tb_name,
        columns=["text_tokenized", "lang", "class"],
        nrows_classes={"geology": rows_class,
                       "gaming": rows_class,
                       "ukraine": rows_class}
    )
    return str(pd.util.hash_pandas_object(controller.df).agg(lambda x: hash(sum(x))))


@callback(
    Output("table", "children"),
    Input("placeholder-load-df", "children"),

)
def update_table(hash_df):
    if int(hash_df) > 0:
        table = controller.get_data_table(max_words=125)
        mapping_columns = {"text_tokenized": "Tweet[:125] (tokenized)", "lang": "Language",
                           "class": "Category"}
        return html.Div([
            dash_table.DataTable(
                fixed_rows={'headers': True},
                style_table={'height': 400, "margin-bottom": "15px"},  # defaults to 500
                style_header={
                    'textAlign': 'center'},
                style_cell={
                    'textAlign': 'center'
                },
                style_cell_conditional=[
                    {'if': {'column_id': 'lang'},
                     'width': '11%'},
                ],
                virtualization=True,
                data=table.to_dict('records'),
                columns=[{"name": mapping_columns[i], "id": i} for i in table.columns \
                         if i != "topic_class" and i != "words_count"]
            )
        ])


# Descriptive Window
@callback(
    Output('hist-plot-tweets', 'figure'),
    Input("placeholder-load-df", "children")
)
def histogram_tweets_count(hash_df):
    if int(hash_df) > 0:
        df = controller.df
        try:
            df["words_count"] = controller.get_tweets_count(df)
            fig = px.histogram(df, x="words_count", color="class")
            fig.update_layout(
                xaxis_title="Number of Words",  # Add x-axis label
                yaxis_title="Number of Tweets"  # Add y-axis label
            )
        except:
            fig = px.histogram(df, x="words_count", color="class")
            fig.update_layout(
                xaxis_title="Number of Words",  # Add x-axis label
                yaxis_title="Number of Tweets"  # Add y-axis label
            )
        return fig


@callback(
    Output('bar-plot-tweets', 'figure'),
    Input('slider-selection-ntokens', 'value'),
    Input("placeholder-load-df", "children")
)
def bar_words_count(number_tokens, hash_df):
    if int(hash_df) > 0:
        df = controller.df
        try:
            fig = px.bar(
                pd.DataFrame(controller.get_token_count(df, pos=int(number_tokens)),
                             columns=["Word", "Count"]),
                x="Word", y="Count",
                title=f"Top {number_tokens} words in Tweets Corpus"
            )
            fig.update_layout(
                xaxis_title="Word",  # Add x-axis label
                yaxis_title="Number of Tweets"  # Add y-axis label
            )
        except:
            fig = px.bar(
                pd.DataFrame(controller.get_token_count(df, pos=int(number_tokens)),
                             columns=["Word", "Count"]),
                x="Word", y="Count",
                title=f"Top {number_tokens} words in Tweets Corpus"
            )
            fig.update_layout(
                xaxis_title="Word",  # Add x-axis label
                yaxis_title="Number of Tweets"  # Add y-axis label
            )
        return fig


# -------------------------------------------------------------------------------------------------


# Model Window
@callback(
    [Output("slider-selection-topic", "marks"),
     Output("slider-selection-topic", "max"),
     Output("slider-selection-topic", "value")],
    [Input("slider-selection-number-topics", "value")]
)
def update_topics_index(number_topics):
    marks = {i: {"label": f"Topic {i}", 'style': {'font-size': '14px'}}
             for i in range(number_topics - 1)}
    return marks, number_topics - 1, 0


@callback(
    Output("placeholder-load-model", 'children'),
    Input('dropdown-selection-models', 'value'),
    Input('slider-selection-number-topics', 'value'),
    Input("placeholder-load-df", "children")

)
def load_model(model_name,
               number_topics,
               hash_df):
    if int(hash_df) > 0:
        df = controller.df
        if controller.if_exists([model_name, number_topics, hash_df]) == None:
            controller.load_model(
                df,
                model_name=models_d[model_name],
                hyperparams=controller.get_hyperparams(
                    models_d[model_name],
                    number_topics=number_topics
                ),
                min_count=1
            )
            controller.wrapper.fit_transform(all_decomposition=True)
            controller.set_model([model_name, number_topics, hash_df])
        else:
            controller.get_model([model_name, number_topics, hash_df])
        params = [model_name, number_topics, hash_df]
        return str(hash("".join([str(params[i]) for i in range(len(params))])))


@callback(
    Output('hist-plot-tweets-by-topics', 'figure'),
    Input('dropdown-selection-models', 'value'),
    Input('slider-selection-number-topics', 'value'),
    Input("placeholder-load-model", 'children'),
    Input("placeholder-load-df", "children"),

)
def histogram_tweets_count_by_topics(
        model_name,
        number_topics,
        hash_true_load_model,
        hash_df
):
    if int(hash_df) > 0 and \
            controller.verify_model_loaded(
                hash_true_load_model,
                [model_name, number_topics, hash_df]
            ):
        df = controller.df
        df["topic_class"] = controller.wrapper.get_topics()
        df["words_count"] = controller.get_tweets_count(df) if "words_count" not in df.columns \
            else df["words_count"]
        try:
            fig = px.histogram(df, x="words_count", color="topic_class",
                               title="Distribution of number of words conditioned by the category",
                               labels={"topic_class": "Topic"})
            fig.update_layout(
                xaxis_title="Number of Words",  # Add x-axis label
                yaxis_title="Number of Tweets"  # Add y-axis label
            )

        except:
            fig = px.histogram(df, x="words_count", color="topic_class",
                               title="Distribution of number of words conditioned by the category",
                               labels={"topic_class": "Topic"})
            fig.update_layout(
                xaxis_title="Number of Words",  # Add x-axis label
                yaxis_title="Number of Tweets"  # Add y-axis label
            )
        return fig


@callback(
    Output('bar-plot-tweets-by-topics', 'figure'),
    [Input('dropdown-selection-models', 'value'),
     Input('slider-selection-topic', 'value'),
     Input('slider-selection-number-topics', 'value'),
     Input('slider-selection-ntokens-by-topics', 'value'),
     Input("placeholder-load-model", 'children'),
     Input("placeholder-load-df", "children")]

)
def bar_words_count_by_topics(model_name,
                              topic, number_topics,
                              number_tokens,
                              hash_true_load_model,
                              hash_df):
    if int(hash_df) > 0 and \
            controller.verify_model_loaded(hash_true_load_model,
                                           [model_name, number_topics,
                                            hash_df]):
        df = controller.df
        PMI_arr, weights_by_topics = controller.get_pmi_scores(number_tokens)
        weights_by_topics = weights_by_topics[topic]
        try:
            fig = px.bar(
                pd.DataFrame(weights_by_topics,
                             columns=["Key Words", "Weight"]),
                x="Key Words", y="Weight",
                title=f"Top {number_tokens} key words in Tweets Corpus for Topic {topic}"
            )
        except:
            fig = px.bar(
                pd.DataFrame(weights_by_topics,
                             columns=["Key Words", "Weight"]),
                x="Key Words", y="Weight",
                title=f"Top {number_tokens} key words in Tweets Corpus for Topic {topic}"
            )
        return fig


@callback(
    Output('image_wc_by_topics', 'src'),
    Input('dropdown-selection-models', 'value'),
    Input('slider-selection-number-topics', 'value'),
    Input("slider-selection-topic", "value"),
    Input("placeholder-load-model", 'children'),
    Input("placeholder-load-df", "children")

)
def create_wc_by_topics(model_name,
                        number_topics,
                        topic,
                        hash_true_load_model,
                        hash_df):
    if int(hash_df) > 0 and \
            controller.verify_model_loaded(hash_true_load_model,
                                           [model_name,
                                            number_topics,
                                            hash_df]):
        img = BytesIO()
        controller.plot_wc(dict_freq=controller.get_dict_freq_word(topic)) \
            .save(img, format='PNG')
        return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())


@callback(
    Output('scatter-plot-tweets-by-topics', 'figure'),
    Input("dropdown-selection-reduc-method", "value"),
    Input('dropdown-selection-models', 'value'),
    Input('slider-selection-number-topics', 'value'),
    Input("placeholder-load-model", 'children'),
    Input("placeholder-load-df", "children")

)
def scatter_tweets_embeddings_by_topics(reduc_method,
                                        model_name,
                                        number_topics,
                                        hash_true_load_model,
                                        hash_df):
    if int(hash_df) > 0 and \
            controller.verify_model_loaded(hash_true_load_model,
                                           [model_name,
                                            number_topics,
                                            hash_df]):

        tweets_embeddings = controller.get_tweets_embeddings_reduc(
            params=[reduc_method_d[reduc_method],
                    model_name,
                    number_topics,
                    hash_df]
        )
        # columns --> ["0","1","topic_class"]
        try:
            fig = px.scatter(tweets_embeddings, x="0", y="1", color="topic_class",
                             title="Representation of Tweets", labels={"topic_class": "Topic"})
            fig.update_layout(
                xaxis_title="1ère composante",  # Add x-axis label
                yaxis_title="2ème composante"  # Add y-axis label
            )
        except:
            fig = px.scatter(tweets_embeddings, x="0", y="1", color="topic_class",
                             title="Representation of Tweets", labels={"topic_class": "Topic"})
            fig.update_layout(
                xaxis_title="First Component",  # Add x-axis label
                yaxis_title="Second Component"  # Add y-axis label
            )
        return fig


@callback(
    Output('metrics', 'children'),
    Input('dropdown-selection-models', 'value'),
    Input('slider-selection-number-topics', 'value'),
    Input("placeholder-load-model", 'children'),
    Input("placeholder-load-df", "children"),

)
def metrics_table(model_name, number_topics, hash_true_load_model, hash_df):
    if int(hash_df) > 0 and controller. \
            verify_model_loaded(hash_true_load_model, [model_name, number_topics, hash_df]):
        table = controller.get_metrics()
        return html.Div([
            dash_table.DataTable(
                fixed_rows={'headers': True},
                style_table={'height': 400, "margin-bottom": "15px"},  # defaults to 500
                style_header={
                    'textAlign': 'center'},
                style_cell={
                    'textAlign': 'center'
                },
                style_cell_conditional=[
                    {'if': {'column_id': 'lang'},
                     'width': '11%'},
                ],
                virtualization=True,
                data=table.to_dict('records'),
                columns=[{"name": i, "id": i} for i in table.columns]
            )
        ])


# ------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    app.run_server(port=8055, debug=True)
