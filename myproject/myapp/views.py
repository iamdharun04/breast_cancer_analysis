from django.shortcuts import render
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from factor_analyzer import FactorAnalyzer
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Use the non-interactive Agg backend
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from io import BytesIO
import base64
from scipy.stats import chi2_contingency


def home(request):
    return render(request, 'myapp/home.html', {'message': 'Welcome to Our Breast Cancer Analysis Project'})


def predict(request):
    data = load_data()
    states = data['States/UT'].dropna().unique().tolist()
    context = {'states': states}

    if request.method == 'POST':
        state = request.POST.get('state')
        predictions, graph = predict_values(state, data)
        context.update({'predictions': predictions, 'graph': graph, 'state': state})

    return render(request, 'myapp/predict.html', context)


def predicta(request):
    return render(request, 'myapp/predict.html', {'message': 'This is another prediction view.'})


def load_data():
    return pd.read_csv("C:/Users/Admin/Downloads/Breast_cancer_india(2016-2021).csv")


def predict_values(state, data):
    state_data = data[data['States/UT'] == state]
    years = [int(year) for year in data.columns[1:]]
    values = state_data.iloc[0, 1:].values.flatten()

    # Ensure we have enough data points for linear regression
    if len(years) < 2:
        return [], None  # Return empty predictions and no graph if insufficient data

    # Fit linear regression model
    model = LinearRegression()
    model.fit(np.array(years).reshape(-1, 1), values.reshape(-1, 1))

    # Predict future values
    future_years = np.array([2022, 2023, 2024, 2025]).reshape(-1, 1)
    future_values = model.predict(future_years).flatten()

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.scatter(years, values, color='blue', label='Historical Data')
    plt.plot(years + list(future_years.flatten()), np.concatenate([values, future_values]), 'r-',
             label='Projected Data')
    plt.xlabel('Year')
    plt.ylabel('Values')
    plt.title(f"Linear Regression and Projection for {state}")
    plt.legend()
    plt.grid(True)

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    graph = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()

    return future_values.tolist(), graph


def perform_kmeans(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=3, random_state=0)  # Example with 3 clusters, adjust as needed
    kmeans.fit(data_scaled)
    labels = kmeans.labels_
    return labels


def analyze_clusters(data):
    # Ensure the data is numeric before groupby and mean calculations
    numeric_data = data.select_dtypes(include=[np.number])
    cluster_means = numeric_data.groupby('Cluster').mean().mean(axis=1)
    return {
        0: "Lower incidence rates.",
        1: "Moderate incidence rates.",
        2: "Higher incidence rates.",
        'means': cluster_means.to_dict()  # Include actual mean values if needed
    }


def plot_cluster_distribution(data):
    cluster_counts = data['Cluster'].value_counts().sort_index()
    plt.figure(figsize=(8, 6))
    cluster_counts.plot(kind='bar', color=['blue', 'orange', 'green'])
    plt.title('Distribution of K-means Clusters')
    plt.xlabel('Cluster')
    plt.ylabel('Number of States')
    plt.xticks(ticks=[0, 1, 2], labels=['Cluster 0', 'Cluster 1', 'Cluster 2'])
    plt.grid(True)

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    graph = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()

    return graph


def clustering_view(request):
    try:
        data = load_data()

        # Assuming 'Cluster' is a column in your data and the data contains numeric values for clustering
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            return render(request, 'myapp/error.html', {'error_message': 'No numeric data available for clustering.'})

        # Perform K-means clustering
        kmeans_labels = perform_kmeans(numeric_data)
        data['Cluster'] = kmeans_labels

        # Analyze clusters
        cluster_descriptions = analyze_clusters(data)

        # Prepare data for template
        states_clusters = zip(data['States/UT'], kmeans_labels)

        # Plot cluster distribution
        cluster_distribution_graph = plot_cluster_distribution(data)

        context = {
            'states_clusters': states_clusters,
            'cluster_descriptions': cluster_descriptions,
            'cluster_distribution_graph': cluster_distribution_graph
        }

        return render(request, 'myapp/clustering.html', context)

    except Exception as e:
        return render(request, 'myapp/error.html', {'error_message': str(e)})


def perform_pca(data):
    # Assuming the first column contains state names or other non-numeric data
    data = data.select_dtypes(include=[np.number])  # Ensure only numeric data is processed
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    pca = PCA(n_components=2)
    components = pca.fit_transform(data_scaled)
    explained_variance = pca.explained_variance_ratio_
    return components, explained_variance


def create_plot(components):
    plt.figure(figsize=(8, 6))
    plt.scatter(components[:, 0], components[:, 1], alpha=0.5)
    plt.title('PCA Results: Scatter Plot of Principal Components')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    graph = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return graph


def pca_view(request):
    try:
        data = load_data()
        components, explained_variance = perform_pca(data)
        graph = create_plot(components)
        context = {
            'explained_variance': explained_variance,
            'graph': graph,
            'has_variance': explained_variance.size > 0
        }
        return render(request, 'myapp/pca.html', context)
    except Exception as e:
        return render(request, 'myapp/error.html', {'error_message': str(e)})


def perform_factor_analysis(data):
    # Extract numeric data for factor analysis
    numeric_data = data.select_dtypes(include=[np.number])

    # Drop any rows with missing values
    numeric_data.dropna(inplace=True)

    # Perform factor analysis using FactorAnalyzer library
    fa = FactorAnalyzer(rotation=None)
    fa.fit(numeric_data)

    # Get factor loadings
    factor_loadings = fa.loadings_

    # Convert factor loadings to DataFrame for easier manipulation
    factor_loadings_df = pd.DataFrame(factor_loadings, index=numeric_data.columns)

    return factor_loadings_df


def factor_analysis_view(request):
    try:
        data = load_data()
        factor_analysis_results = perform_factor_analysis(data)
        context = {
            'factor_analysis_results': factor_analysis_results,
            'has_results': not factor_analysis_results.empty
        }
        return render(request, 'myapp/factor_analysis.html', context)
    except Exception as e:
        return render(request, 'myapp/error.html', {'error_message': str(e)})


def perform_chi_square_test(data, variable1, variable2):
    # Remove the "Total" row if it exists
    data = data[data['States/UT'] != 'Total']

    # Check if variable1 is in the DataFrame columns
    if variable1 not in data.columns:
        raise ValueError(f"Variable '{variable1}' is not found in the DataFrame columns.")

    # Check if all elements of variable2 are in the DataFrame columns
    if not all(v in data.columns for v in variable2):
        missing_vars = [v for v in variable2 if v not in data.columns]
        raise ValueError(f"One or more variables in {missing_vars} are not found in the DataFrame columns.")

    # Extract relevant columns for the contingency table
    contingency_data = data.set_index(variable1)[variable2]

    # Perform the chi-square test
    chi2_stat, p_value, _, _ = chi2_contingency(contingency_data)

    return chi2_stat, p_value


def chi_square_test_view(request):
    try:
        data = load_data()

        # Specify thet categorical variables for the test
        variable1 = 'States/UT'  # Replace 'States/UT' with the actual column name
        variable2 = ['2016', '2017', '2018', '2019', '2020', '2021']

        # Perform Chi-square test
        chi2_stat, p_value = perform_chi_square_test(data, variable1, variable2)

        context = {
            'chi2_stat': chi2_stat,
            'p_value': p_value
        }
        return render(request, 'myapp/chi_square_results.html', context)
    except Exception as e:
        return render(request, 'myapp/error.html', {'error_message': str(e)})

