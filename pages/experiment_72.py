import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import streamlit as st

# Read data from Excel sheet
data = pd.read_csv('./Data_csv/circle_coordinates_71.csv')

# Extracting relevant columns
center_x = data['Center X']
center_y = data['Center Y']

# Combine center_x and center_y into a single numpy array
coordinates = np.column_stack((center_x, center_y))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Extracting relevant columns
center_x = data['Center X']
center_y = data['Center Y']

# Define the dimensions of the screen (adjust as needed)
screen_width = 1920  # Example: Screen width in pixels
screen_height = 1080  # Example: Screen height in pixels

# Create 2D histogram to represent the density of gaze points
heatmap, xedges, yedges = np.histogram2d(center_x, center_y, bins=[screen_width//10, screen_height//10])  # Adjust bin size as needed

# Plot the heatmap
plt.figure(figsize=(10, 8))
plt.imshow(heatmap.T, extent=[0, screen_width, 0, screen_height], origin='lower', cmap='hot')
plt.colorbar(label='Frequency')
plt.title('Frequency of Focus')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)
st.pyplot(plt.gcf())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Assuming 'Time' is the column containing the time stamps
data['Time (s)'] = pd.to_datetime(data['Time (s)'])

# Define screen dimensions
screen_width = 1920  # Example: Screen width in pixels
screen_height = 1080  # Example: Screen height in pixels

# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0, screen_width)
ax.set_ylim(0, screen_height)
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_title('Trajectory Animation')

# Initialize empty scatter plot for data points
scatter = ax.scatter([], [], c='blue', marker='o', alpha=0.5)

# Update function to animate the scatter plot
def update(frame):
    # Get data points up to current frame
    current_data = data[data['Time'] <= data['Time'].iloc[frame]]
    x = current_data['Center X']
    y = current_data['Center Y']
    scatter.set_offsets(np.column_stack((x, y)))
    return scatter,

# Create animation
ani = FuncAnimation(fig, update, frames=len(data), interval=0.01)  # Adjust interval as needed

# Show animation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Extracting relevant columns
center_x = data['Center X']
center_y = data['Center Y']

# Create 2D histogram (heatmap)
heatmap, xedges, yedges = np.histogram2d(center_x, center_y, bins=50)

# Plot heatmap
plt.figure(figsize=(10, 8))
plt.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower', cmap='hot')
plt.colorbar(label='Frequency')
plt.title('User Gaze Heatmap')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)
st.pyplot(plt.gcf())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# Extracting relevant columns
center_x = data['Center X']
center_y = data['Center Y']

# Combine center_x and center_y into a single numpy array
coordinates = np.column_stack((center_x, center_y))

# Define number of clusters
n_clusters = 5  # Adjust as needed

# Apply K-means clustering
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(coordinates)

# Get cluster centers
cluster_centers = kmeans.cluster_centers_

# Plot data points and cluster centers
plt.scatter(center_x, center_y, c=kmeans.labels_, cmap='viridis', alpha=0.5)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', color='red', label='Cluster Centers')
plt.title('K-means Clustering for Region Detection')
plt.xlabel('Center X')
plt.ylabel('Center Y')
plt.legend()
st.pyplot(plt.gcf())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
from matplotlib.colors import ListedColormap

# Read data from Excel sheet

# Extracting relevant columns
center_x = data['Center X']
center_y = data['Center Y']

# Combine center_x and center_y into a single numpy array
coordinates = np.column_stack((center_x, center_y))

# Apply Mean Shift clustering
meanshift = MeanShift(bandwidth=2)
meanshift.fit(coordinates)

# Create a colormap with labels
colors = np.array(['blue', 'red', 'green', 'orange'])  # Adjust as needed
labels = np.array(['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'])  # Adjust as needed
cmap = ListedColormap(colors)

# Plot data points
plt.scatter(center_x, center_y, c=meanshift.labels_, cmap=cmap, alpha=0.5)
plt.title('Mean Shift Clustering for Region Detection')
plt.xlabel('Center X')
plt.ylabel('Center Y')
# Add legend with labels
plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label) for color, label in zip(colors, labels)])
st.pyplot(plt.gcf())

import pandas as pd
import matplotlib.pyplot as plt

# Read data from Excel sheet

# Extracting relevant columns
time = data['Time (s)']
center_x = data['Center X']
center_y = data['Center Y']

# Plot trajectory
plt.figure(figsize=(8, 6))
plt.plot(center_x, center_y, marker='o', markersize=5, linestyle='-', color='b')
plt.title('User Gaze Trajectory')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)
st.pyplot(plt.gcf())

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors



# Extracting relevant columns
time = data['Time (s)']
center_x = data['Center X']
center_y = data['Center Y']

# Normalize time to range [0, 1] for color mapping
normalized_time = (time - time.min()) / (time.max() - time.min())

# Define colormap
cmap = plt.cm.viridis

# Plot trajectory with varying marker color and size based on time
plt.figure(figsize=(8, 6))
for i in range(len(center_x)):
    plt.scatter(center_x[i], center_y[i], c=cmap(normalized_time[i]), s=100, alpha=0.5)

# Add colorbar to show time
norm = mcolors.Normalize(vmin=time.min().timestamp(), vmax=time.max().timestamp())
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap))
cbar.set_label('Time (s)')

plt.title('User Gaze Trajectory')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

# Read data from Excel sheet

# Extracting relevant columns
time = data['Time (s)']
center_x = data['Center X']
center_y = data['Center Y']

# Normalize time to range [0, 1] for color mapping
normalized_time = (time - time.min()) / (time.max() - time.min())

# Define colormap
cmap = plt.cm.viridis

# Plot trajectory with varying marker color and size based on time
plt.figure(figsize=(8, 6))
for i in range(len(center_x)):
    plt.scatter(center_x[i], center_y[i], c=cmap(normalized_time[i]), s=100, alpha=0.5)

# Plot lines connecting consecutive points
for i in range(len(center_x) - 1):
    plt.plot([center_x[i], center_x[i + 1]], [center_y[i], center_y[i + 1]], color='gray', alpha=0.5)

# Add colorbar to show time
norm = mcolors.Normalize(vmin=time.min().timestamp(), vmax=time.max().timestamp())
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap))
cbar.set_label('Time (s)')

plt.title('User Gaze Trajectory')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)
st.pyplot(plt.gcf())

# Apply Hierarchical Clustering
Z = linkage(coordinates, method='ward')

# Plot dendrogram
plt.figure(figsize=(10, 5))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')
dendrogram(Z)
st.pyplot(plt.gcf())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from matplotlib.colors import ListedColormap


# Extracting relevant columns
center_x = data['Center X']
center_y = data['Center Y']

# Combine center_x and center_y into a single numpy array
coordinates = np.column_stack((center_x, center_y))

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.1, min_samples=5)
dbscan.fit(coordinates)

# Create a colormap with labels
colors = np.array(['blue', 'red', 'green', 'orange'])  # Adjust as needed
labels = np.array(['Cluster 1', 'Cluster 2', 'Cluster 3', 'Noise'])  # Adjust as needed
cmap = ListedColormap(colors)

# Plot data points
plt.scatter(center_x, center_y, c=dbscan.labels_, cmap=cmap, alpha=0.5)
plt.title('DBSCAN Clustering for Region Detection')
plt.xlabel('Center X')
plt.ylabel('Center Y')
# Add legend with labels
plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label) for color, label in zip(colors, labels)])
st.pyplot(plt.gcf())
# But the ./Data_csv/circle_coordinates_72.csv should be n-1 of the experiment number, that is the above code is for experiment_73