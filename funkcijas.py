def sataisit_grafiku(df_test, x_df, y_df, ticker, pieaugums, dienas):
    import plotly.graph_objects as go
    # Create Plotly figure
    fig = go.Figure()
    
    # Plot close price as line
    fig.add_trace(go.Scatter(
        x=x_df, y=y_df,
        mode='lines', name='Close Price', line=dict(color='gray')))
    
    # Up predictions: green ↑ triangle
    fig.add_trace(go.Scatter(
        x=df_test[df_test['Predicted_up'] == 1].Datetime,
        y=df_test[df_test['Predicted_up'] == 1]['Close'],
        mode='markers',
        name=f'Predict +{pieaugums}%',
        marker=dict(symbol='triangle-up', color='green', size=10)
    ))    
   
    # Layout
    fig.update_layout(
        title=f'''Model Predictions of "{ticker}" at least +{pieaugums}% after {dienas} days''',
        xaxis_title='Datetime',
        yaxis_title='Price',
        legend=dict(x=0, y=1),
        template='plotly_white',
        height=800,
        width=1400
    )
    
    # Set dark theme
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='Black')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='Black')
    fig.update_layout(
        plot_bgcolor='rgb(30,30,30)',
        paper_bgcolor='rgb(15,15,15)',
        font=dict(color='white'),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            bordercolor='black'
        )
    )
    
    fig.show()
