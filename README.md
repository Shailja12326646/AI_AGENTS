# AMAZON ML CHALLENGE 2025
Smart Product Pricing: Methodology 
This document outlines the methodology, feature engineering techniques, and modeling approach used to 
predict product prices based on the provided dataset. Our approach focuses on robust feature extraction from 
textual data, combined with numerical and categorical features, to train a high-performance Gradient 
Boosting model. 
Methodology Used 
Our end-to-end pipeline follows a structured, multi-stage approach: 
1. Data Preprocessing and Parsing: The raw text_block data was the primary source of information. 
We began by parsing this unstructured text using regular expressions to extract core product 
attributes: Item Name, Bullet Points, Value (e.g., weight, volume), and Unit. 
2. Comprehensive Feature Engineering: We created a rich feature set by combining and transforming 
the parsed data. This involved: 
o Textual Feature Extraction: Advanced NLP techniques were used to extract brand names, 
product descriptions, and categorical tags. 
o Numerical Feature Creation: New numerical features were derived from existing data to 
capture relationships like price-per-unit and total pack value. 
o Data Cleaning: Missing values were handled by either dropping columns with low 
completion rates or imputing the mean for key numerical features. 
3. Model Training and Evaluation: A LightGBM (LGBM) Regressor was chosen for its efficiency 
and high performance on tabular and sparse data. The final feature set, combining numerical, 
categorical, and TF-IDF vectorized text data, was used to train the model. Performance was 
evaluated using the competition's Symmetric Mean Absolute Percentage Error (SMAPE) metric. 
Feature Engineering Techniques Applied 
A variety of techniques were employed to transform the raw data into meaningful features for the model. 
Text-Based Features 
• Text Cleaning and Consolidation: item_name and bullets were combined into a single corpus 
(clean_text), which was then lowercased, and stripped of special characters and extra whitespace. 
• Brand Name Extraction and Normalization: 
o A primary Brand Name was extracted from the item_name using regex to identify 
capitalized words at the start of the string. 
o To handle variations (e.g., "Stonewall Kitchen", "Stonewall"), we used the rapidfuzz library 
to group similar brand names under a single canonical name (brand_clean). 
• Smart Product Name Extraction: We used the spaCy library to perform Part-of-Speech (POS) 
tagging on the item name. By extracting key nouns, adjectives, and proper nouns (while removing 
brand names and units), we created a concise and descriptive smart_product_name feature. 
• Automated Categorization: An extensive dictionary of keywords and regex patterns was created to 
automatically classify products into categories (e.g., coffee, snack, sauce). For products that 
remained unclassified, we used KMeans clustering on their TF-IDF vectors to create automated 
category labels (auto_cat_*). 
• TF-IDF Vectorization: The clean_text and item_name columns were vectorized using 
TfidfVectorizer to convert the text into a numerical format, capturing the importance of different 
words. We used 500 features for clean_text and 300 for item_name. 
• Binary Tagging: A diet_tag feature was created to flag products with dietary keywords like 
"organic," "gluten-free," or "vegan." 
Numerical and Categorical Features 
• Derived Price Metrics: We created price_per_unit by dividing price by value and pack_value by 
multiplying value by pack_size. 
• Metadata Features: word_count (from clean_text) and num_bullets were calculated to quantify the 
amount of descriptive information available. 
• Popularity Score: A brand_popularity feature was created by calculating the frequency of each 
brand in the dataset. 
• Encoding: StandardScaler was applied to all numerical features, and OneHotEncoder was used 
for categorical features like unit and category_guess. 
Model Architecture/Algorithms Selected 
We selected the LightGBM (LGBM) Regressor, a gradient boosting framework known for its speed and 
accuracy. 
• Why LightGBM? It's highly efficient with large datasets and handles a mix of sparse (TF-IDF) and 
dense (numerical/encoded) features effectively. Its leaf-wise growth strategy often leads to better 
accuracy than traditional level-wise approaches. 
• Hyperparameters: The model was trained with n_estimators=1000 and a learning_rate=0.05, 
providing a good balance between training speed and performance. 
Other Relevant Information 
• Data Cleaning: Columns with a high percentage of missing values after initial processing (e.g., 
country_or_origin, flavor_or_variant) were dropped to reduce noise. Remaining missing numerical 
values were imputed with the column mean. 
• Performance: On a local 80/20 train-test split, the model achieved a SMAPE score of 6.94%, 
indicating a strong predictive capability. 
• Holistic Approach: The model's success is attributed to its ability to learn from a wide array of 
engineered features, capturing signals from raw text, brand popularity, product quantity, and derived 
categories.
