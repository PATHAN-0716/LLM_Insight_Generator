# LLM_Insight_Generator
An interactive data analysis tool leveraging Google's Gemini LLM and LangChain to generate insights and visualizations from any tabular (CSV) data using natural language queries

## Project Overview

This repository hosts the **LLM Powered Tabular Insight Generator**, an interactive tool designed to extract meaningful insights and generate visualizations from any tabular data (CSV format) using natural language queries. Leveraging the power of Google's Gemini Large Language Models via LangChain, this project enables users to perform sophisticated data analysis without writing a single line of code.

Simply upload your CSV, ask a question about your data, and watch the AI agent generate answers, perform calculations, or even create dynamic plots like bar charts, line plots, and scatter plots.

## Features

* **Natural Language Interaction:** Query your dataset using everyday language.
* **Dynamic Data Analysis:** Get summaries, aggregations, and filtered results.
* **Automated Data Visualization:** Request various plot types (bar, line, scatter, histogram, boxplot, pie, heatmap) which are generated on the fly.
* **Flexible Data Loading:** Designed to work with *any* CSV file provided by the user.
* **Google Gemini API Integration:** Powered by advanced Large Language Models for intelligent responses.
* **LangChain Integration:** Utilizes the LangChain Pandas DataFrame Agent for robust data interaction.
* **Secure API Key Handling:** Employs Google Colab's secret manager for secure API key storage.

## How to Run

### Option 1: Open in Google Colab (Recommended for ease of use)

This is the quickest way to get started and interact with the LLM Powered Tabular Insight Generator.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/LLM-Powered-Tabular-Insight-Generator/blob/main/notebooks/tabular_insight_generator.ipynb)

**Steps:**

1.  **Click the "Open in Colab" badge above.** This will open a copy of the notebook in your Google Colab environment.
2.  **(Optional but Recommended) Save a copy to Drive:** Go to `File` > `Save a copy in Drive` if you want to save your own editable version.
3.  **Set up Google API Key:**
    * In Google Colab, click the **key icon** (🔑) on the left sidebar (this opens the "Secrets" panel).
    * Click `+ New secret`.
    * Set the `Name` to `GOOGLE_API_KEY`.
    * Paste your actual Google API key into the `Value` field.
    * Make sure to enable `Notebook access` for this secret (toggle the switch next to it).
4.  **Upload your CSV Data:**
    * Click the **folder icon** (📁) on the left sidebar (this opens the "Files" panel).
    * Click the **"Upload to session storage" icon** (page with an upward arrow) and select your CSV file (e.g., `Superstore.csv` or any other dataset).
    * *(Note: This project is designed for any CSV file. `data/sample/Superstore.csv` is included in this repository as a demonstration dataset.)*
5.  **Run All Cells:** Go to `Runtime` > `Run all` in the Colab menu.
6.  **Enter CSV File Name:** When prompted, enter the exact name of your uploaded CSV file (e.g., `Superstore.csv`).
7.  **Start Asking Questions!** Once the agent is initialized, you can type your questions about your data in the prompt.

### Option 2: Local Setup (for Developers/Advanced Users)

If you prefer to run the project locally on your machine.

**Prerequisites:**

* Python 3.9+
* Git installed
* A Google Gemini API Key

**Steps:**

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/LLM-Powered-Tabular-Insight-Generator.git](https://github.com/YOUR_USERNAME/LLM-Powered-Tabular-Insight-Generator.git)
    cd LLM-Powered-Tabular-Insight-Generator
    ```
    *Replace `YOUR_USERNAME` with your GitHub username.*
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    # .\venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Place your CSV data:**
    * Place your CSV file (e.g., `Superstore.csv`) into the `data/sample/` directory, or any other location you specify when running the notebook.
5.  **Set your Google API Key:**
    * Create a `.env` file in the root directory of the project (`LLM-Powered-Tabular-Insight-Generator/`).
    * Add your API key to this file:
        ```
        GOOGLE_API_KEY="YOUR_ACTUAL_GOOGLE_API_KEY"
        ```
        *(Ensure `.env` is included in your `.gitignore` file to prevent accidental commits of your key).*
6.  **Run the notebook:**
    ```bash
    jupyter notebook notebooks/tabular_insight_generator.ipynb
    ```
    Follow the prompts within the notebook to load data and interact.

## Project Structure

```
LLM-Powered-Tabular-Insight-Generator/
├── notebooks/
│   └── tabular_insight_generator.ipynb   # The core notebook with the LLM agent and interaction logic.
├── data/
│   └── sample/
│       └── Superstore.csv                # A sample dataset to get started quickly.
├── src/
│   └── utils.py                          # (Optional) Example: Place refactored data processing or LLM setup functions here.
├── img/
│   └── demo_screenshot.png               # Screenshot of the project in action or a generated plot.
├── .gitignore                            # Specifies files and directories to be ignored by Git (e.g., API keys, temporary files).
├── README.md                             # This comprehensive project overview, usage instructions, and details.
├── requirements.txt                      # Lists all necessary Python dependencies for the project.
└── LICENSE                               # Defines the project's licensing (e.g., MIT License).
```


## Dependencies

All required Python packages are listed in `requirements.txt`. These include:

* `pandas`
* `matplotlib`
* `seaborn`
* `langchain`
* `google-generativeai`
* `langchain-google-genai`
* `langchain-experimental`
* `python-dotenv` (for local `.env` file loading, not strictly needed for Colab secrets)

## Example Questions (Try these!)

Below are some examples of questions you can ask the AI agent to get insights or visualizations. Remember, you can ask about any column in your loaded dataset!

**General Tips for Asking Questions:**

* **Be clear and specific:** Refer to your exact column names (e.g., `sales`, `product_id`, `category`).
* **Ask for aggregations:** E.g., "What is the `total` of `column_X`?" or "What is the `average` of `column_Y` by `column_Z`?"
* **Ask for filtering:** E.g., "Show me data where `status` is 'completed'."
* **Request plots:** E.g., "Show me a `bar chart` of `total_revenue` by `city`." or "Create a `scatter plot` of `feature_x` vs `feature_y`."
* **Common Plot Types:** Bar chart, Line plot, Scatter plot, Histogram, Boxplot, Pie chart, Heatmap.

**Example Questions:**

1.  "What are the top 5 `product_names` by `sales`?"
2.  "Show me the average `profit` for each `segment`."
3.  "Create a line plot showing the `total_sales` trend over `order_date`."
4.  "Generate a bar chart of `sales` by `region`."
5.  "Is there a correlation between `sales` and `profit`? Show me a scatter plot."
6.  "What is the distribution of `discount` values? Show a histogram."
7.  "Which `ship_mode` has the highest average `shipping_cost`?"
8.  "Plot the number of orders per `customer_id` for the top 10 customers."
9.  "Show me the `total_profit` for `technology` `category` in `California`."
10. "Generate a heatmap of average `sales` by `category` and `sub_category`."

---

**Challenging Example Questions (Try these if you're feeling adventurous!):**

1.  Which product 'Sub-Category' had the largest percentage drop in sales from 2016 to 2017, considering only sub-categories that had sales in both years?
2.  Is there any noticeable correlation between the 'Discount' applied to an order item and the 'Profit' generated from that item? If so, describe it and identify the 'Sub-Category' where this relationship is most prominent (either positively or negatively).
3.  Identify any 'Customer ID' who consistently purchased across all four main 'Region's ('East', 'West', 'Central', 'South') and provide their total sales and total profit across all their orders.
4.  For orders where the 'Ship Mode' was 'Same Day', what is the average profit margin, and how does this compare to the average profit margin for orders shipped with 'Standard Class'? Does the faster shipping justify the difference?
5.  Beyond simple totals, can you find the 'Product Name' that, despite having high sales volume (e.g., top 10% in quantity sold), consistently results in negative profit? List its average profit and discount.

---

## License

This project is licensed under the [MIT License](LICENSE).
