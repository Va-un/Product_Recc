```markdown
# BigBasket Product Recommendation System

A content-based recommendation system built for BigBasket products using machine learning techniques to suggest similar items based on product features and categories.

## Overview

This project implements a product recommendation system that analyzes product characteristics including category, sub-category, brand, and type to provide personalized product suggestions. The system uses cosine similarity on TF-IDF vectorized product features to find and recommend similar products.

## Features

- **Content-Based Filtering**: Recommends products based on item features and characteristics
- **Interactive Web Interface**: Built with Streamlit for easy user interaction
- **Smart Search**: Find products with partial name matching
- **Customizable Results**: Adjust the number of recommended products (1-40)
- **Product Analytics**: Display product details including price, rating, and description

## Installation

1. Clone the repository:
```
git clone 
cd bigbasket-recommendation-system
```

2. Install required dependencies:
```
pip install -r requirements.txt
```

## Dependencies

- **NumPy**: Numerical computing library
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning library for vectorization and similarity calculations
- **regex**: Regular expression operations for text processing
- **Streamlit**: Web app framework for the user interface

## Dataset

The system uses a CSV file named `BigBasket Products.csv` containing product information with the following columns:
- `product`: Product name
- `category`: Product category
- `sub_category`: Product sub-category  
- `brand`: Product brand
- `sale_price`: Current selling price
- `market_price`: Original market price
- `type`: Product type/variant
- `rating`: Customer rating
- `description`: Product description

## Usage

### Running the Streamlit App

```
streamlit run Main.py
```

### Using the Recommendation System

1. **Enter Product Name**: Type the name of a product you're interested in
2. **Select from Suggestions**: Choose from the dropdown of matching products
3. **Set Number of Recommendations**: Use the slider to specify how many similar products to display (0-40)
4. **Get Recommendations**: Click "Similar products" to see results

### Programmatic Usage

```
# Import the recommendation function
from Main import find

# Get 5 similar products to "Cadbury Perk - Chocolate Bar"
recommendations = find('Cadbury Perk - Chocolate Bar', 5)
print(recommendations)
```

## How It Works

1. **Data Preprocessing**: 
   - Removes null values from the dataset
   - Cleans and normalizes text data (categories, brands, types)
   - Creates feature combinations for each product

2. **Feature Engineering**:
   - Combines category, sub_category, brand, and type into a single feature vector
   - Uses CountVectorizer to convert text features into numerical format

3. **Similarity Calculation**:
   - Computes cosine similarity between all products
   - Creates a similarity matrix for efficient recommendation lookup

4. **Recommendation Generation**:
   - Finds products with highest similarity scores
   - Returns top N similar products with their details

## File Structure

```
├── Main.py                          # Streamlit web application
├── recommendation-system.ipynb      # Jupyter notebook with detailed analysis
├── requirements.txt                 # Python dependencies
├── dataset-cover.jpg               # Project cover image
└── BigBasket Products.csv          # Product dataset (required)
```

## Example Output

When searching for "Cadbury Perk - Chocolate Bar", the system returns similar products like:
- Nutties Chocolate Pack (Cadbury)
- 5 Star Chocolate Bar (Cadbury) 
- Dairy Milk Silk Hazelnut Chocolate Bar (Cadbury)
- Dark Milk Chocolate Bar (Cadbury)

Each recommendation includes product name, category, brand, price, rating, and description.

## Technical Details

- **Algorithm**: Content-based filtering using cosine similarity
- **Vectorization**: CountVectorizer with English stop words removal
- **Similarity Metric**: Cosine similarity for measuring product feature overlap
- **Framework**: Streamlit for web interface, Pandas/NumPy for data processing



```

