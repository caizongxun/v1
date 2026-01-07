import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report


class ZigZagAnalysisDashboard:
    """
    Interactive dashboard for ZigZag ML model analysis and visualization
    """
    
    def __init__(self, predictions, true_labels, confidences, timestamps=None):
        self.predictions = predictions
        self.true_labels = true_labels
        self.confidences = confidences
        self.timestamps = timestamps or np.arange(len(predictions))
        self.class_names = ['HH', 'HL', 'LH', 'LL', 'No Pattern']
    
    def plot_confusion_matrix(self):
        """Create confusion matrix heatmap"""
        cm = confusion_matrix(self.true_labels, self.predictions)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=self.class_names,
            y=self.class_names,
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 12}
        ))
        
        fig.update_layout(
            title='Confusion Matrix - ZigZag Pattern Predictions',
            xaxis_title='Predicted Label',
            yaxis_title='True Label',
            height=600,
            width=700
        )
        
        return fig
    
    def plot_confidence_distribution(self):
        """Plot confidence score distribution by class"""
        data = []
        for i in range(5):
            mask = self.true_labels == i
            if mask.sum() > 0:
                data.append(go.Box(
                    y=self.confidences[mask],
                    name=self.class_names[i],
                    boxmean='sd'
                ))
        
        fig = go.Figure(data=data)
        fig.update_layout(
            title='Confidence Score Distribution by Pattern Type',
            yaxis_title='Confidence Score',
            xaxis_title='Pattern Type',
            height=500,
            showlegend=True
        )
        
        return fig
    
    def plot_accuracy_by_class(self):
        """Calculate and plot per-class accuracy"""
        accuracies = []
        sample_counts = []
        
        for i in range(5):
            mask = self.true_labels == i
            if mask.sum() > 0:
                correct = (self.predictions[mask] == self.true_labels[mask]).sum()
                accuracy = correct / mask.sum()
                accuracies.append(accuracy)
                sample_counts.append(mask.sum())
            else:
                accuracies.append(0)
                sample_counts.append(0)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=self.class_names,
            y=accuracies,
            name='Accuracy',
            marker_color='steelblue',
            text=[f'{a:.1%}' for a in accuracies],
            textposition='auto'
        ))
        
        fig.update_layout(
            title='Per-Class Accuracy',
            yaxis_title='Accuracy',
            xaxis_title='Pattern Type',
            height=500,
            yaxis_range=[0, 1]
        )
        
        return fig
    
    def plot_prediction_timeline(self, window_size: int = 100):
        """Plot rolling accuracy over time"""
        rolling_accuracy = []
        rolling_confidence = []
        
        for i in range(len(self.predictions) - window_size):
            window_pred = self.predictions[i:i+window_size]
            window_true = self.true_labels[i:i+window_size]
            window_conf = self.confidences[i:i+window_size]
            
            accuracy = (window_pred == window_true).mean()
            rolling_accuracy.append(accuracy)
            rolling_confidence.append(window_conf.mean())
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Rolling Accuracy', 'Rolling Confidence'),
            specs=[[{'secondary_y': False}], [{'secondary_y': False}]]
        )
        
        fig.add_trace(
            go.Scatter(y=rolling_accuracy, name='Rolling Accuracy', 
                      line=dict(color='steelblue'), mode='lines'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(y=rolling_confidence, name='Rolling Confidence',
                      line=dict(color='coral'), mode='lines'),
            row=2, col=1
        )
        
        fig.update_yaxes(title_text='Accuracy', row=1, col=1, range=[0, 1])
        fig.update_yaxes(title_text='Confidence', row=2, col=1, range=[0, 1])
        fig.update_xaxes(title_text='Sample Index', row=2, col=1)
        
        fig.update_layout(
            title=f'Performance Metrics Over Time (window={window_size})',
            height=700,
            showlegend=True
        )
        
        return fig
    
    def plot_class_distribution(self):
        """Plot class distribution in predictions vs true labels"""
        pred_counts = pd.Series(self.predictions).value_counts().sort_index()
        true_counts = pd.Series(self.true_labels).value_counts().sort_index()
        
        # Ensure all classes are represented
        for i in range(5):
            if i not in pred_counts.index:
                pred_counts[i] = 0
            if i not in true_counts.index:
                true_counts[i] = 0
        
        pred_counts = pred_counts.sort_index()
        true_counts = true_counts.sort_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=self.class_names,
            y=true_counts.values,
            name='True Labels',
            marker_color='steelblue'
        ))
        
        fig.add_trace(go.Bar(
            x=self.class_names,
            y=pred_counts.values,
            name='Predicted Labels',
            marker_color='coral'
        ))
        
        fig.update_layout(
            title='Class Distribution Comparison',
            yaxis_title='Count',
            xaxis_title='Pattern Type',
            barmode='group',
            height=500
        )
        
        return fig
    
    def plot_prediction_errors(self):
        """Plot where model makes mistakes"""
        errors = self.predictions != self.true_labels
        error_rate_by_class = {}
        
        for i in range(5):
            mask = self.true_labels == i
            if mask.sum() > 0:
                error_rate = errors[mask].mean()
                error_rate_by_class[self.class_names[i]] = error_rate
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(error_rate_by_class.keys()),
                y=list(error_rate_by_class.values()),
                marker_color='indianred',
                text=[f'{v:.1%}' for v in error_rate_by_class.values()],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title='Error Rate by True Class',
            yaxis_title='Error Rate',
            xaxis_title='Pattern Type',
            height=500,
            yaxis_range=[0, 1]
        )
        
        return fig
    
    def plot_confidence_vs_accuracy(self):
        """Analyze relationship between confidence and accuracy"""
        errors = self.predictions != self.true_labels
        
        df_data = pd.DataFrame({
            'Confidence': self.confidences,
            'Correct': ~errors,
            'Class': [self.class_names[i] for i in self.true_labels]
        })
        
        fig = px.scatter(
            df_data,
            x='Confidence',
            y='Correct',
            color='Class',
            marginal_x='histogram',
            title='Confidence vs Prediction Correctness',
            labels={'Correct': 'Prediction Correct'},
            height=600
        )
        
        fig.update_yaxes(type='category')
        
        return fig
    
    def create_dashboard(self, output_file: str = 'zigzag_dashboard.html'):
        """Create comprehensive dashboard with all plots"""
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Confusion Matrix',
                'Per-Class Accuracy',
                'Class Distribution',
                'Error Rate by Class',
                'Confidence Distribution',
                'Predictions Over Time'
            ),
            specs=[
                [{'type': 'heatmap'}, {'type': 'bar'}],
                [{'type': 'bar'}, {'type': 'bar'}],
                [{'type': 'box'}, {'type': 'scatter'}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # 1. Confusion Matrix
        cm = confusion_matrix(self.true_labels, self.predictions)
        fig.add_trace(
            go.Heatmap(z=cm, x=self.class_names, y=self.class_names,
                      colorscale='Blues', showscale=False),
            row=1, col=1
        )
        
        # 2. Per-class Accuracy
        accuracies = []
        for i in range(5):
            mask = self.true_labels == i
            if mask.sum() > 0:
                accuracies.append((self.predictions[mask] == self.true_labels[mask]).mean())
            else:
                accuracies.append(0)
        
        fig.add_trace(
            go.Bar(x=self.class_names, y=accuracies, marker_color='steelblue',
                  showlegend=False),
            row=1, col=2
        )
        
        # 3. Class Distribution
        true_counts = pd.Series(self.true_labels).value_counts().sort_index()
        pred_counts = pd.Series(self.predictions).value_counts().sort_index()
        
        fig.add_trace(
            go.Bar(x=self.class_names, y=true_counts.values, name='True',
                  marker_color='steelblue', showlegend=False),
            row=2, col=1
        )
        
        # 4. Error Rate
        errors = self.predictions != self.true_labels
        error_rates = []
        for i in range(5):
            mask = self.true_labels == i
            if mask.sum() > 0:
                error_rates.append(errors[mask].mean())
            else:
                error_rates.append(0)
        
        fig.add_trace(
            go.Bar(x=self.class_names, y=error_rates, marker_color='indianred',
                  showlegend=False),
            row=2, col=2
        )
        
        # 5. Confidence by class
        for i in range(5):
            mask = self.true_labels == i
            if mask.sum() > 0:
                fig.add_trace(
                    go.Box(y=self.confidences[mask], name=self.class_names[i],
                          showlegend=i==0),
                    row=3, col=1
                )
        
        # 6. Rolling accuracy
        window_size = 50
        rolling_acc = []
        for i in range(len(self.predictions) - window_size):
            acc = (self.predictions[i:i+window_size] == self.true_labels[i:i+window_size]).mean()
            rolling_acc.append(acc)
        
        fig.add_trace(
            go.Scatter(y=rolling_acc, mode='lines', name='Rolling Acc',
                      line=dict(color='steelblue'), showlegend=False),
            row=3, col=2
        )
        
        fig.update_layout(
            title_text='ZigZag ML Model - Comprehensive Analysis Dashboard',
            height=1200,
            width=1400,
            showlegend=False
        )
        
        fig.update_yaxes(title_text='True Class', row=1, col=1)
        fig.update_yaxes(title_text='Accuracy', row=1, col=2, range=[0, 1])
        fig.update_yaxes(title_text='Count', row=2, col=1)
        fig.update_yaxes(title_text='Error Rate', row=2, col=2, range=[0, 1])
        fig.update_yaxes(title_text='Confidence', row=3, col=1)
        fig.update_yaxes(title_text='Accuracy', row=3, col=2, range=[0, 1])
        
        fig.write_html(output_file)
        print(f"Dashboard saved to {output_file}")
        
        return fig


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    n_samples = 500
    
    # Simulated data
    true_labels = np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.25, 0.25, 0.2, 0.2, 0.1])
    predictions = true_labels.copy()
    
    # Add some errors for realism
    error_indices = np.random.choice(n_samples, int(0.15*n_samples), replace=False)
    predictions[error_indices] = np.random.choice([0, 1, 2, 3, 4], len(error_indices))
    
    confidences = np.random.beta(7, 2, n_samples)  # Biased towards higher confidence
    
    # Create dashboard
    dashboard = ZigZagAnalysisDashboard(predictions, true_labels, confidences)
    dashboard.create_dashboard('zigzag_analysis.html')
    
    print("Dashboard created successfully!")
