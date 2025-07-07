# Cell 1: Import Libraries and Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, r2_score, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder  # Added LabelEncoder import
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")

# Cell 2: Task 1a - Load Data
try:
    languages_df = pd.read_csv('languages.tsv', sep='\t')
    forms_df = pd.read_csv('forms.tsv', sep='\t')
    print(f"Languages: {languages_df.shape}, Forms: {forms_df.shape}")
    print("\nLanguages columns:", languages_df.columns.tolist())
    print("Forms columns:", forms_df.columns.tolist())
    print("\nFirst few rows of languages_df:")
    print(languages_df.head())
    print("\nFirst few rows of forms_df:")
    print(forms_df.head())
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please ensure the files 'languages.tsv' and 'forms.tsv' are in the current directory")

# Cell 3: Task 1b - Aggregate and Merge Data
# Aggregate forms data by language
forms_agg = forms_df.groupby('isocode').agg({
    'wordLength': 'mean',
    'longestClusterLength': 'mean',
    'vowConsRatio': 'mean'
}).round(3)

# Rename columns as specified
forms_agg.columns = ['avgLength', 'avgCluster', 'avgVowRatio']
forms_agg.reset_index(inplace=True)

# Merge with languages dataframe
main_df = languages_df.merge(forms_agg, on='isocode', how='left')
print(f"Main dataframe shape: {main_df.shape}")
print("\nMain dataframe columns:", main_df.columns.tolist())
print("\nFirst few rows of merged data:")
print(main_df.head())

# Cell 4: Task 1c - Find Min/Max Values
numerical_vars = ['inventorySize', 'avgLength', 'avgCluster', 'avgVowRatio']
print("Extreme values for each numerical variable:")
print("=" * 50)

for var in numerical_vars:
    if var in main_df.columns:
        min_idx = main_df[var].idxmin()
        max_idx = main_df[var].idxmax()
        min_val = main_df.loc[min_idx, var]
        max_val = main_df.loc[max_idx, var]
        min_lang = main_df.loc[min_idx, 'name']
        max_lang = main_df.loc[max_idx, 'name']
        
        print(f"{var}:")
        print(f"  Min: {min_lang} ({min_val:.3f})")
        print(f"  Max: {max_lang} ({max_val:.3f})")
        print()

# Cell 5: Task 1d - Create Pairplot
plt.figure(figsize=(12, 10))
if all(col in main_df.columns for col in numerical_vars):
    plot_data = main_df[numerical_vars + ['family']].copy()
    g = sns.pairplot(plot_data, hue='family', diag_kind='hist', 
                     plot_kws={'alpha': 0.7}, diag_kws={'alpha': 0.7})
    g.fig.suptitle('Pairwise Distributions of Sound System Properties by Language Family', 
                   y=1.02, fontsize=14)
    plt.tight_layout()
    plt.show()
else:
    print("Some required columns are missing for pairplot")

# Correlation matrix
corr_matrix = main_df[numerical_vars].corr()
print("\nCorrelation matrix:")
print(corr_matrix.round(3))

# Visualize correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.3f')
plt.title('Correlation Matrix of Sound System Properties')
plt.tight_layout()
plt.show()

# Cell 6: Task 2a - Choose Statistical Test
print("Task 2: Testing whether Families Differ in Inventory Size")
print("=" * 60)
print("\nResearch Question: Do sound inventory sizes differ between language families?")
print("\nApproach: One-way ANOVA (Analysis of Variance)")
print("\nJustification:")
print("- We have one continuous dependent variable (inventory size)")
print("- We have one categorical independent variable (language family) with 4 groups")
print("- We want to test if the means of inventory size differ across groups")
print("- ANOVA is appropriate for comparing means across multiple groups")

# Cell 7: Task 2b - Check ANOVA Assumptions
anova_data = main_df[['inventorySize', 'family']].dropna()
print(f"\nANOVA analysis using {anova_data.shape[0]} observations")

# Descriptive statistics by family
print("\nDescriptive statistics by family:")
family_stats = anova_data.groupby('family')['inventorySize'].describe()
print(family_stats)

# Check normality assumption (Shapiro-Wilk test for each group)
print("\nChecking normality assumption (Shapiro-Wilk test):")
for family in anova_data['family'].unique():
    family_data = anova_data[anova_data['family'] == family]['inventorySize']
    if len(family_data) >= 3:  # Shapiro-Wilk requires at least 3 observations
        stat, p_value = stats.shapiro(family_data)
        print(f"{family}: W = {stat:.4f}, p = {p_value:.4f}")
    else:
        print(f"{family}: Too few observations for normality test")

# Check homogeneity of variances (Levene's test)
family_groups = [group['inventorySize'].values for name, group in anova_data.groupby('family')]
levene_stat, levene_p = stats.levene(*family_groups)
print(f"\nLevene's test for equal variances:")
print(f"F = {levene_stat:.4f}, p = {levene_p:.4f}")

if levene_p > 0.05:
    print("✓ Homogeneity of variances assumption appears to be met")
else:
    print("✗ Homogeneity of variances assumption may be violated")

# Cell 8: Task 2c - Perform ANOVA
print("\nPerforming One-way ANOVA:")
print("-" * 30)

# Fit ANOVA model
model = ols('inventorySize ~ C(family)', data=anova_data).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print("ANOVA Results:")
print(anova_table)

# Extract key statistics
f_stat = anova_table.loc['C(family)', 'F']
p_value = anova_table.loc['C(family)', 'PR(>F)']
alpha = 0.05

print(f"\nKey Results:")
print(f"F-statistic: {f_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Alpha level: {alpha}")

# Interpret results
if p_value < alpha:
    print(f"\n✓ SIGNIFICANT RESULT: p = {p_value:.4f} < {alpha}")
    print("We reject the null hypothesis. There are significant differences in inventory size between language families.")
    
    # Post-hoc analysis if significant
    try:
        tukey = pairwise_tukeyhsd(endog=anova_data['inventorySize'], 
                                groups=anova_data['family'], alpha=0.05)
        print("\nPost-hoc analysis (Tukey HSD):")
        print(tukey)
    except Exception as e:
        print(f"Could not perform post-hoc analysis: {e}")
        
else:
    print(f"\n✗ NON-SIGNIFICANT RESULT: p = {p_value:.4f} >= {alpha}")
    print("We fail to reject the null hypothesis. There are no significant differences in inventory size between language families.")

# Effect size (eta-squared)
ss_between = anova_table.loc['C(family)', 'sum_sq']
ss_total = anova_table['sum_sq'].sum()
eta_squared = ss_between / ss_total
print(f"\nEffect size (η²): {eta_squared:.4f}")

# Cell 9: Task 3 - Linear Regression Models Setup
print("Task 3: Attempting to Predict Average Word Length")
print("=" * 50)

# Define variables
X_vars = ['inventorySize', 'avgCluster', 'avgVowRatio']
y_var = 'avgLength'

# Prepare modeling data
model_data = main_df[X_vars + [y_var]].dropna()
print(f"Modeling data: {model_data.shape[0]} observations")
print(f"Predictors: {X_vars}")
print(f"Target: {y_var}")

# Check for missing values
print("\nMissing values:")
print(model_data.isnull().sum())

# Cell 10: Task 3 - Single Predictor Models
print("\nSingle Predictor Models:")
print("-" * 30)

single_models = {}
for predictor in X_vars:
    X = sm.add_constant(model_data[predictor])
    y = model_data[y_var]
    model = sm.OLS(y, X).fit()
    single_models[predictor] = model
    
    print(f"{predictor}:")
    print(f"  R² = {model.rsquared:.4f}")
    print(f"  Adj R² = {model.rsquared_adj:.4f}")
    print(f"  AIC = {model.aic:.2f}")
    print(f"  p-value = {model.f_pvalue:.4f}")
    print()

# Cell 11: Task 3 - Two Predictor Models
print("Two Predictor Models:")
print("-" * 30)

two_models = {}
for pred_pair in combinations(X_vars, 2):
    model_name = ' + '.join(pred_pair)
    X = sm.add_constant(model_data[list(pred_pair)])
    y = model_data[y_var]
    model = sm.OLS(y, X).fit()
    two_models[model_name] = model
    
    print(f"{model_name}:")
    print(f"  R² = {model.rsquared:.4f}")
    print(f"  Adj R² = {model.rsquared_adj:.4f}")
    print(f"  AIC = {model.aic:.2f}")
    print(f"  p-value = {model.f_pvalue:.4f}")
    print()

# Cell 12: Task 3 - Three Predictor Model and Best Model Selection
print("Three Predictor Model:")
print("-" * 30)

X = sm.add_constant(model_data[X_vars])
y = model_data[y_var]
full_model = sm.OLS(y, X).fit()

print(f"Full model (all predictors):")
print(f"  R² = {full_model.rsquared:.4f}")
print(f"  Adj R² = {full_model.rsquared_adj:.4f}")
print(f"  AIC = {full_model.aic:.2f}")
print(f"  p-value = {full_model.f_pvalue:.4f}")

# Model comparison
print("\nModel Comparison Summary:")
print("=" * 50)

all_models = {**single_models, **two_models, 'Full Model': full_model}

# Create comparison table
comparison_data = []
for name, model in all_models.items():
    comparison_data.append({
        'Model': name,
        'R²': model.rsquared,
        'Adj R²': model.rsquared_adj,
        'AIC': model.aic,
        'p-value': model.f_pvalue
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('Adj R²', ascending=False)
print(comparison_df.round(4))

# Identify best model
best_model_name = comparison_df.iloc[0]['Model']
best_model = all_models[best_model_name]

print(f"\nBest Model: {best_model_name}")
print(f"Adjusted R² = {best_model.rsquared_adj:.4f}")
print(f"This model explains {best_model.rsquared_adj*100:.1f}% of the variance in average word length")

# Determine predictors for sklearn comparison
if best_model_name == 'Full Model':
    best_predictors = X_vars
elif '+' in best_model_name:
    best_predictors = best_model_name.split(' + ')
else:
    best_predictors = [best_model_name]

print(f"Best predictors: {best_predictors}")

# Show detailed results for best model
print(f"\nDetailed Results for Best Model:")
print(best_model.summary())

# Cell 13: Task 4 - Scikit-learn Model Comparison
print("Task 4: Comparing Linear Regression Models")
print("=" * 50)

# Prepare data for sklearn
X = model_data[best_predictors]
y = model_data['avgLength']

# Standardize features for regularized models
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Using predictors: {best_predictors}")
print(f"Data shape: {X.shape}")

# Cell 14: Task 4 - Fit Different Models
# Regular Linear Regression
lr = LinearRegression().fit(X, y)
lr_r2 = lr.score(X, y)

# Lasso Regression - find best alpha
print("\nTesting different alpha values for Lasso:")
alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
best_lasso_r2 = -np.inf
best_lasso_alpha = 0.1

for alpha in alphas:
    lasso_temp = Lasso(alpha=alpha, max_iter=2000).fit(X_scaled, y)
    r2 = lasso_temp.score(X_scaled, y)
    print(f"Alpha = {alpha}: R² = {r2:.4f}")
    if r2 > best_lasso_r2:
        best_lasso_r2 = r2
        best_lasso_alpha = alpha

print(f"Best Lasso alpha: {best_lasso_alpha}")

# Fit final models
lasso = Lasso(alpha=best_lasso_alpha, max_iter=2000).fit(X_scaled, y)
elastic = ElasticNet(alpha=best_lasso_alpha, l1_ratio=0.5, max_iter=2000).fit(X_scaled, y)

# Model comparison
print(f"\nModel Performance Comparison:")
print("-" * 35)
print(f"Linear Regression:  R² = {lr.score(X, y):.4f}")
print(f"Lasso Regression:   R² = {lasso.score(X_scaled, y):.4f}")
print(f"ElasticNet:         R² = {elastic.score(X_scaled, y):.4f}")

# Analysis of coefficients
print(f"\nCoefficient Analysis:")
print("-" * 25)
print("Linear Regression coefficients:")
for i, pred in enumerate(best_predictors):
    print(f"  {pred}: {lr.coef_[i]:.4f}")

print("Lasso coefficients:")
for i, pred in enumerate(best_predictors):
    print(f"  {pred}: {lasso.coef_[i]:.4f}")

print("ElasticNet coefficients:")
for i, pred in enumerate(best_predictors):
    print(f"  {pred}: {elastic.coef_[i]:.4f}")

# Interpretation
print(f"\nInterpretation:")
if lasso.score(X_scaled, y) < lr.score(X, y):
    print("- Regularized models (Lasso, ElasticNet) perform worse than standard linear regression")
    print("- This suggests that regularization is penalizing the model too heavily")
    print("- The dataset may be too small to benefit from regularization")
    print("- Standard linear regression is preferred for this dataset")

# Cell 15: Task 5a - Prepare Data for Logistic Regression
def prepare_data_for_prediction(df):
    """
    Prepare the data for logistic regression prediction
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Convert phonVowLength to binary (1 for True, 0 for False)
    data['vowel_length_binary'] = data['phonVowLength'].astype(int)
    
    # Handle any missing values
    print("Missing values per column:")
    print(data[['inventorySize', 'avgLength', 'avgCluster', 'avgVowRatio', 'family', 'vowel_length_binary']].isnull().sum())
    
    # Remove rows with missing values in key columns
    data = data.dropna(subset=['inventorySize', 'avgLength', 'avgCluster', 'avgVowRatio', 'family', 'vowel_length_binary'])
    
    return data

def encode_categorical_features(data):
    """
    Encode categorical variables (language family)
    """
    # Label encode the family variable
    le = LabelEncoder()
    data['family_encoded'] = le.fit_transform(data['family'])
    
    # Create dummy variables for family (alternative approach)
    family_dummies = pd.get_dummies(data['family'], prefix='family')
    data = pd.concat([data, family_dummies], axis=1)
    
    return data, le

def train_logistic_regression(main_df):
    """
    Train and evaluate logistic regression model for phonemic vowel length prediction
    """
    print("=" * 60)
    print("PHONEMIC VOWEL LENGTH PREDICTION - LOGISTIC REGRESSION")
    print("=" * 60)
    
    # Prepare data
    data = prepare_data_for_prediction(main_df)
    print(f"\nDataset shape after cleaning: {data.shape}")
    
    # Encode categorical features
    data, label_encoder = encode_categorical_features(data)
    
    # Define predictor variables
    predictor_cols = ['inventorySize', 'avgLength', 'avgCluster', 'avgVowRatio', 'family_encoded']
    
    # Alternative: Use dummy variables for family instead
    family_cols = [col for col in data.columns if col.startswith('family_')]
    predictor_cols_with_dummies = ['inventorySize', 'avgLength', 'avgCluster', 'avgVowRatio'] + family_cols
    
    # Prepare features and target
    X_encoded = data[predictor_cols]
    X_dummies = data[predictor_cols_with_dummies]
    y = data['vowel_length_binary']
    
    print(f"\nTarget variable distribution:")
    print(f"Languages with phonemic vowel length: {y.sum()} ({y.mean():.2%})")
    print(f"Languages without phonemic vowel length: {(1-y).sum()} ({(1-y.mean()):.2%})")
    
    # Split data
    X_train_enc, X_test_enc, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train_dum, X_test_dum, _, _ = train_test_split(
        X_dummies, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler_enc = StandardScaler()
    X_train_enc_scaled = scaler_enc.fit_transform(X_train_enc)
    X_test_enc_scaled = scaler_enc.transform(X_test_enc)
    
    scaler_dum = StandardScaler()
    X_train_dum_scaled = scaler_dum.fit_transform(X_train_dum)
    X_test_dum_scaled = scaler_dum.transform(X_test_dum)
    
    # Train models
    print("\n" + "="*40)
    print("MODEL 1: LABEL ENCODED FAMILY")
    print("="*40)
    
    # Model 1: Label encoded family
    lr_encoded = LogisticRegression(random_state=42, max_iter=1000)
    lr_encoded.fit(X_train_enc_scaled, y_train)
    
    # Predictions
    y_pred_enc = lr_encoded.predict(X_test_enc_scaled)
    y_pred_proba_enc = lr_encoded.predict_proba(X_test_enc_scaled)[:, 1]
    
    # Evaluate Model 1
    print(f"Accuracy: {accuracy_score(y_test, y_pred_enc):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba_enc):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_enc))
    
    # Cross-validation
    cv_scores_enc = cross_val_score(lr_encoded, X_train_enc_scaled, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validation scores: {cv_scores_enc}")
    print(f"Mean CV accuracy: {cv_scores_enc.mean():.4f} (+/- {cv_scores_enc.std() * 2:.4f})")
    
    # Feature importance (coefficients)
    feature_names_enc = predictor_cols
    coefficients_enc = lr_encoded.coef_[0]
    
    print("\nFeature Coefficients (Model 1):")
    for name, coef in zip(feature_names_enc, coefficients_enc):
        print(f"{name}: {coef:.4f}")
    
    print("\n" + "="*40)
    print("MODEL 2: DUMMY ENCODED FAMILY")
    print("="*40)
    
    # Model 2: Dummy encoded family
    lr_dummy = LogisticRegression(random_state=42, max_iter=1000)
    lr_dummy.fit(X_train_dum_scaled, y_train)
    
    # Predictions
    y_pred_dum = lr_dummy.predict(X_test_dum_scaled)
    y_pred_proba_dum = lr_dummy.predict_proba(X_test_dum_scaled)[:, 1]
    
    # Evaluate Model 2
    print(f"Accuracy: {accuracy_score(y_test, y_pred_dum):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba_dum):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_dum))
    
    # Cross-validation
    cv_scores_dum = cross_val_score(lr_dummy, X_train_dum_scaled, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validation scores: {cv_scores_dum}")
    print(f"Mean CV accuracy: {cv_scores_dum.mean():.4f} (+/- {cv_scores_dum.std() * 2:.4f})")
    
    # Feature importance for dummy model
    feature_names_dum = predictor_cols_with_dummies
    coefficients_dum = lr_dummy.coef_[0]
    
    print("\nTop Feature Coefficients (Model 2):")
    coef_df = pd.DataFrame({
        'feature': feature_names_dum,
        'coefficient': coefficients_dum,
        'abs_coefficient': np.abs(coefficients_dum)
    }).sort_values('abs_coefficient', ascending=False)
    
    print(coef_df.head(10))
    
    # Create visualizations
    create_visualizations(y_test, y_pred_proba_enc, y_pred_proba_dum, 
                         feature_names_enc, coefficients_enc, 
                         coef_df, data)
    
    return lr_encoded, lr_dummy, scaler_enc, scaler_dum, label_encoder

def create_visualizations(y_test, y_pred_proba_enc, y_pred_proba_dum, 
                         feature_names_enc, coefficients_enc, coef_df, data):
    """
    Create visualizations for model evaluation
    """
    plt.figure(figsize=(15, 12))
    
    # ROC Curves
    plt.subplot(2, 3, 1)
    fpr_enc, tpr_enc, _ = roc_curve(y_test, y_pred_proba_enc)
    fpr_dum, tpr_dum, _ = roc_curve(y_test, y_pred_proba_dum)
    
    plt.plot(fpr_enc, tpr_enc, label=f'Label Encoded (AUC = {roc_auc_score(y_test, y_pred_proba_enc):.3f})')
    plt.plot(fpr_dum, tpr_dum, label=f'Dummy Encoded (AUC = {roc_auc_score(y_test, y_pred_proba_dum):.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Feature coefficients (Model 1)
    plt.subplot(2, 3, 2)
    colors = ['red' if x < 0 else 'blue' for x in coefficients_enc]
    plt.barh(feature_names_enc, coefficients_enc, color=colors)
    plt.xlabel('Coefficient Value')
    plt.title('Feature Coefficients (Label Encoded)')
    plt.grid(True, alpha=0.3)
    
    # Top features (Model 2)
    plt.subplot(2, 3, 3)
    top_features = coef_df.head(8)
    colors = ['red' if x < 0 else 'blue' for x in top_features['coefficient']]
    plt.barh(top_features['feature'], top_features['coefficient'], color=colors)
    plt.xlabel('Coefficient Value')
    plt.title('Top Feature Coefficients (Dummy Encoded)')
    plt.grid(True, alpha=0.3)
    
    # Prediction probability distribution
    plt.subplot(2, 3, 4)
    plt.hist(y_pred_proba_enc[y_test == 0], bins=20, alpha=0.7, label='No Vowel Length', color='red')
    plt.hist(y_pred_proba_enc[y_test == 1], bins=20, alpha=0.7, label='Has Vowel Length', color='blue')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('Prediction Probability Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Target distribution by family
    plt.subplot(2, 3, 5)
    family_vowel_length = data.groupby('family')['vowel_length_binary'].agg(['count', 'mean']).reset_index()
    family_vowel_length = family_vowel_length[family_vowel_length['count'] >= 3]  # Only families with 3+ languages
    
    plt.bar(range(len(family_vowel_length)), family_vowel_length['mean'])
    plt.xlabel('Language Family')
    plt.ylabel('Proportion with Vowel Length')
    plt.title('Vowel Length by Family')
    plt.xticks(range(len(family_vowel_length)), family_vowel_length['family'], rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Correlation with target
    plt.subplot(2, 3, 6)
    numeric_cols = ['inventorySize', 'avgLength', 'avgCluster', 'avgVowRatio']
    correlations = [data[col].corr(data['vowel_length_binary']) for col in numeric_cols]
    colors = ['red' if x < 0 else 'blue' for x in correlations]
    plt.barh(numeric_cols, correlations, color=colors)
    plt.xlabel('Correlation with Vowel Length')
    plt.title('Feature Correlations with Target')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
