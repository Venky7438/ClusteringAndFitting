import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

 
def read_and_preprocess_data(file_path):
    """
            Read and preprocess a dataset for clustering.

            Parameters:
            - file_path (str): The path to the CSV file containing the dataset.

            Returns:
            - data (pd.DataFrame): The preprocessed DataFrame for clustering.
            - columns_for_clustering (list): List of column names selected for clustering.
            - transpose_data (pd.DataFrame): The transposed and preprocessed DataFrame.
            """
    # Read the data
    data = pd.read_csv(file_path)

    # Select relevant columns for clustering
    columns_for_clustering = [
        "Current account balance (% of GDP) [BN.CAB.XOKA.GD.ZS]" ,
        "Foreign direct investment, net inflows (% of GDP) [BX.KLT.DINV.WD.GD.ZS]" ,
        "Foreign direct investment, net outflows (% of GDP) [BM.KLT.DINV.WD.GD.ZS]" ,
        "Personal remittances, received (% of GDP) [BX.TRF.PWKR.DT.GD.ZS]" ,
        "Travel services (% of service exports, BoP) [BX.GSR.TRVL.ZS]" ,
        "Travel services (% of service imports, BoP) [BM.GSR.TRVL.ZS]" ,
        "Transport services (% of service exports, BoP) [BX.GSR.TRAN.ZS]" ,
        "Transport services (% of service imports, BoP) [BM.GSR.TRAN.ZS]" ,
        "Trade in services (% of GDP) [BG.GSR.NFSV.GD.ZS]"
    ]

    # Replace non-numeric values with NaN
    data[columns_for_clustering] = data[columns_for_clustering].\
        replace('..' , pd.NA)

    # Convert columns to numeric
    data[columns_for_clustering] = data[columns_for_clustering].\
        apply(pd.to_numeric , errors='coerce')

    # Impute missing values
    imputer = SimpleImputer(strategy = 'mean')
    data[columns_for_clustering] = imputer.fit_transform(data[columns_for_clustering])

    return data , columns_for_clustering


def perform_kmeans_clustering(data , columns_for_clustering , n_clusters = 4):
    """
            Perform k-means clustering on the specified columns of the dataset.

            Parameters:
            - data (pd.DataFrame): The DataFrame containing the dataset.
            - columns_for_clustering (list): List of column names to be
            used for clustering.
            - n_clusters (int, optional): The number of clusters for
            k-means. Default is 4.

            Returns:
            - data (pd.DataFrame): The DataFrame with an additional
            'cluster' column indicating cluster membership.
            - X_normalized (ndarray): The normalized data used for clustering.
            """
    # Normalize the data
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(data[columns_for_clustering])

    # Apply k-means clustering
    kmeans = KMeans(n_clusters = n_clusters , random_state = 42)
    data['cluster'] = kmeans.fit_predict(X_normalized)

    # Calculate silhouette score
    silhouette_avg = silhouette_score(X_normalized , data['cluster'])
    print(f'Silhouette Score: {silhouette_avg}')

    # Plot cluster membership
    plt.scatter(X_normalized[: , 0] , X_normalized[: , 1] ,
                c = data['cluster'], cmap = 'viridis')
    plt.scatter(kmeans.cluster_centers_[: , 0] ,
                kmeans.cluster_centers_[: , 1], s = 300 , c = 'red' , marker = 'X')
    plt.title('K-Means Clustering' , fontsize = 16)
    plt.xlabel('Current account balance (% of GDP)' , fontsize = 16)
    plt.ylabel('Foreign direct investment, net inflows (% of GDP)' , fontsize = 16)
    plt.show()

    return data, X_normalized


def err_ranges(x , params , covariance , z_value):
    """
            Calculate the lower and upper bounds of the confidence
            interval for each point on the curve.

            Parameters:
            - x (array-like): Independent variable values.
            - params (tuple): Fitted parameters of the model.
            - covariance (ndarray): Covariance matrix of the fitted parameters.
            - z_value (float): Z-value corresponding to the desired confidence level.

            Returns:
            - lower_bound (array-like): Lower bounds of the confidence interval for each point.
            - upper_bound (array-like): Upper bounds of the confidence interval for each point.
            """
    # Calculate lower and upper bounds of the confidence interval for each point on the curve.
    y_fit = func(x , *params)

    # Calculate standard deviation of the fitted values at each point
    std_dev = np.sqrt(np.sum((np.dot(np.vander(x , len(params)) ,
                                     covariance) * np.vander(x , len(params))) ** 2 , axis = 1))

    lower_bound = y_fit - z_value * std_dev
    upper_bound = y_fit + z_value * std_dev
    return lower_bound , upper_bound


def func(x , a , b , c):
    """
            Define the function to fit (low-order polynomial).

            Parameters:
            - x (array-like): Independent variable values.
            - a (float): Coefficient for the quadratic term.
            - b (float): Coefficient for the linear term.
            - c (float): Constant term.

            Returns:
            - array-like: Predicted values based on the quadratic function.
            """
    # Define the function to fit (low-order polynomial)
    return a * x ** 2 + b * x + c


def fit_and_plot_curve_time(data , y_data_column , n_points = 100):
    """
            Fit a curve to the provided data and predict values for
            the year 2023 with confidence intervals.

            Parameters:
            - data (pd.DataFrame): The DataFrame containing the dataset.
            - y_data_column (str): The column name representing the dependent variable.
            - countries_to_predict (list): List of country names for which predictions will be made.
            - n_points (int, optional): Number of points for curve fitting. Default is 100.

            Returns:
            None
            """
    # Convert 'Time_Column' to numeric, assuming it contains strings representing years
    # data['Time'] = data['Time'].copy()
    data['Time'] = pd.to_numeric(data['Time'] , errors = 'coerce')

    # Drop rows with NaN values in 'Time_Column'
    data = data.dropna(subset = ['Time'])

    # Extract time and y_data for curve fitting
    time_data = data['Time'].to_numpy()
    y_data = data[y_data_column].to_numpy()

    # Fit the curve
    params , covariance = curve_fit(func , time_data , y_data)

    # Get the standard deviations of the parameters
    # (square root of the diagonal of the covariance matrix)
    std_dev = np.sqrt(np.diag(covariance))

    # Define the range for predictions
    time_range = np.linspace(min(time_data) , max(time_data) , n_points)

    # Generate predictions and confidence intervals
    y_fit = func(time_range , *params)
    lower_bound , upper_bound = err_ranges(time_range , params , covariance , 1)

    # Plot the data, the fitted curve, and confidence intervals
    plt.scatter(time_data , y_data , label = 'Data')
    plt.plot(time_range , y_fit , label = 'Best Fit')
    plt.fill_between(time_range , lower_bound , upper_bound ,
                     alpha = 0.2 , label = 'Confidence Interval')

    plt.title(f'Curve Fitting foreign direct investment  vs Time)')
    plt.xlabel('Time')
    plt.ylabel('Foreign direct investment')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    file_path = 'e1b6f5a2-fde8-4376-88c2-7791df35b5af_Data.csv'

    # Read and preprocess data
    data , columns_for_clustering = read_and_preprocess_data(file_path)

    # Perform k-means clustering
    data , X_normalized = perform_kmeans_clustering(data , columns_for_clustering)

    # Choose one cluster (you may adjust this)
    selected_cluster = 0
    selected_cluster_data = data[data['cluster'] == selected_cluster]

    # Specify the column for fitting
    y_data_column = "Current account balance (% of GDP) [BN.CAB.XOKA.GD.ZS]"

    # Fit and plot the curve for the specified column
    fit_and_plot_curve_time(selected_cluster_data , y_data_column)
