import plotly.express as px
import pycountry
import pycountry_convert as pcc
from fuzzywuzzy import process
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

# Initialize Spark session with enhanced logging
spark = SparkSession.builder \
    .appName('End to end processing') \
    .config("spark.driver.extraJavaOptions", "-Dlog4j.configuration=file:log4j.properties") \
    .getOrCreate()

spark.sparkContext.setLogLevel("INFO")
print("\n🔵 ========== STARTING SPARK JOB ==========\n")

# ----------------- DATA LOADING -----------------
print("✅ [1/8] Initializing Spark session...")
df = spark.read.csv('input/visa_number_in_japan.csv', header=True, inferSchema=True)

print(f"\n📥 [2/8] Loaded raw data:")
print(f"Rows: {df.count():,}")
print(f"Columns: {len(df.columns)}")
print("Sample data:")
df.show(5, vertical=True, truncate=50)

# ----------------- DATA CLEANING -----------------
print("\n🧹 [3/8] Cleaning column names...")
new_column_names = [
    col_name.replace(' ', '_')
    .replace('/', '')
    .replace('.', '')
    .replace(',', '')
    for col_name in df.columns
]
df = df.toDF(*new_column_names)

print("\n🔍 Cleaned schema:")
df.printSchema()
print("\n📋 Sample data after renaming:")
df.show(5, truncate=False)

# ----------------- DATA FILTERING -----------------
print("\n🚮 [4/8] Filtering data...")
initial_count = df.count()
df = df.dropna(how='all')
df = df.select('year', 'country', 'number_of_issued_numerical')

print(f"Removed {(initial_count - df.count()):,} empty rows")
print("\n📑 Post-filtering schema:")
df.printSchema()
print("\n📊 Sample filtered data:")
df.show(10, truncate=False)

# ----------------- COUNTRY NAME CORRECTION -----------------
print("\n🌍 [5/8] Correcting country names...")

def correct_country_name(name, threshold=85):
    countries = [country.name for country in pycountry.countries]
    corrected_name, score = process.extractOne(name, countries)
    return corrected_name if score >= threshold else name

correct_country_name_udf = udf(correct_country_name, StringType())
df = df.withColumn('country', correct_country_name_udf(df['country']))

print("\n🔍 Country correction samples:")
df.groupBy("country").count().orderBy("count", ascending=False).show(10, truncate=False)

# ----------------- CONTINENT MAPPING -----------------
print("\n🗺️ [6/8] Mapping countries to continents...")

def get_continent_name(country_name):
    try:
        country_code = pcc.country_name_to_country_alpha2(country_name)
        continent_code = pcc.country_alpha2_to_continent_code(country_code)
        return pcc.convert_continent_code_to_continent_name(continent_code)
    except:
        return None

continent_udf = udf(get_continent_name, StringType())
df = df.withColumn('continent', continent_udf(df['country']))

print("\n🌐 Continent distribution:")
df.groupBy("continent").count().orderBy("count", ascending=False).show(truncate=False)

# ----------------- DATA VALIDATION -----------------
print("\n🔎 [7/8] Validating transformed data...")
print("📜 Final schema:")
df.printSchema()

print("\n📊 Sample final data:")
df.show(10, truncate=False)

print("\n🧮 Summary statistics:")
df.describe().show()

# ----------------- VISUALIZATIONS -----------------
print("\n📈 [8/8] Generating visualizations...")
df.createOrReplaceGlobalTempView("japan_visa")

# Continent-wise analysis
df_cont = spark.sql("""
    SELECT year, continent, SUM(number_of_issued_numerical) AS visa_issued
    FROM global_temp.japan_visa
    WHERE continent IS NOT NULL
    GROUP BY year, continent
""")

print("\n📦 Continent aggregation results:")
df_cont.show(10)

df_cont_pd = df_cont.toPandas()
fig = px.bar(df_cont_pd, x='year', y='visa_issued', color='continent', barmode='group')
fig.update_layout(
    title_text="Visa Issuance by Continent (2006-2017)",
    xaxis_title='Year',
    yaxis_title='Visas Issued',
    legend_title='Continent'
)
fig.write_html('output/visa_continent.html')
print("✅ Saved continent visualization")

# Country analysis for 2017
df_country = spark.sql("""
    SELECT country, SUM(number_of_issued_numerical) AS visa_issued
    FROM global_temp.japan_visa
    WHERE country NOT IN ('total', 'others')
    AND year = 2017
    GROUP BY country
    ORDER BY visa_issued DESC
    LIMIT 10
""")

print("\n🏆 Top countries 2017:")
df_country.show(truncate=False)

df_country_pd = df_country.toPandas()
fig = px.bar(df_country_pd, x='country', y='visa_issued', color='country')
fig.update_layout(
    title_text="Top 10 Countries by Visa Issuance (2017)",
    xaxis_title='Country',
    yaxis_title='Visas Issued'
)
fig.write_html('output/top_countries_2017.html')
print("✅ Saved country ranking visualization")

# Geographic visualization
df_geo = spark.sql("""
    SELECT year, country, SUM(number_of_issued_numerical) AS visa_issued
    FROM global_temp.japan_visa
    WHERE country NOT IN ('total', 'others')
    GROUP BY year, country
    ORDER BY year
""")

print("\n🌍 Geographic distribution:")
df_geo.show(10)

df_geo_pd = df_geo.toPandas()
fig = px.choropleth(df_geo_pd,
                    locations='country',
                    locationmode='country names',
                    color='visa_issued',
                    animation_frame='year',
                    range_color=[0, 1000000],
                    title="Visa Issuance by Country Over Time")
fig.write_html('output/geographic_distribution.html')
print("✅ Saved geographic visualization")

# ----------------- DATA EXPORT -----------------
print("\n💾 Saving cleaned data...")
df.write.csv("output/cleaned_visa_data", mode="overwrite", header=True)
print("✅ Cleaned data saved to output/cleaned_visa_data")

# ----------------- JOB COMPLETION -----------------
print("\n🔵 ========== JOB COMPLETED SUCCESSFULLY ==========")
print("📂 Output files:")
print(" - output/visa_continent.html")
print(" - output/top_countries_2017.html")
print(" - output/geographic_distribution.html")
print(" - output/cleaned_visa_data/")

spark.stop()
print("\n🛑 Spark session terminated\n")