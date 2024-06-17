import pandas as pd
from dstapi import DstApi
import matplotlib.pyplot as plt
import ipywidgets as widgets

import pandas as pd
from dstapi import DstApi

class Classdataproject:
    def import_data(self, x):  
        FRKM123 = DstApi(x) #We import the data set FRKM123 from Statistics Denmark which has population projections.
        tabsum = FRKM123.tablesummary(language='en')
        #print("Table Summary:", tabsum) 

        params = FRKM123._define_base_params(language='en') #Defining our parameters
        #print("Parameters:", params)
    
        return FRKM123, params

    def rename(self, FRKM123, params): 
        try:
            # Fetch data using params
            proj_api = FRKM123.get_data(params=params)
            #print("Data Fetch Head:", proj_api.head(5))  # Display the first 5 rows for debugging purposes
            
            # Rename column titles to english
            proj_api.rename(columns={
                'OMRÅDE': 'municipality',
                'ALDER': 'age',
                'KØN': 'sex',
                'TID': 'year',
                'INDHOLD': 'population_projection'
            }, inplace=True)

            # Create dataframe
            df = pd.DataFrame(proj_api)
            
            # We remove the municipality Christiansø as the population value is 0 for all years:
            df = df[df['municipality'] != 'Christiansø']
            
            return df
        
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
        
    def process_data(self, df):
        '''We do not want to analyze gender differences.
            We therefore wish to remove this column. 
            First, we create a new column and then we create a variable that that sums up the population projection of men and women for the given municipality, age and year, and fill in the new column. 
            When we have the projections - regardless of gender - we want to delete the 'sex' column. First, we need to delete half of the observations. We delete the rows where sex = Men, such that we only have one row for every municipality, age and year.
            After that, we drop the 'sex' column from the dataframe.'''
        try:
            # Create an empty column for the population projections for both genders and call it 'pop_proj_all'.
            df['pop_proj_all'] = ""
            
            # Summarize the values of population_projection if municipality, age, and year equal each other, respectively.
            def sumval(group):
                if len(group) > 1:
                    return group.sum()
                else:
                    return group.values[0]
            
            df['pop_proj_all'] = df.groupby(['municipality', 'age', 'year'])['population_projection'].transform(sumval)
            
            # Drop observations with men to remove duplicate observations
            df = df[df['sex'] != 'Men']
            
            # Drop the 'sex' and 'population_projection' columns altogether
            columns_to_drop = ['sex', 'population_projection']
            df = df.drop(columns_to_drop, axis=1)
            
            '''We now move on to the 'age' column. We want to create age groups to sort the data. 
                First, we clean up the column by first removing observations called 'Age, total', and make the ages into integers instead of strings.
                Then, we create our age groups, and a variable that sums up all ages into those bins. 
                We again create a new column for it and call it 'prop_proj_age'.'''

            # Drop the value that is not an age
            df = df[df['age'] != 'Age, total']
            
            # Split up the string, such that we only keep the numbers, i.e., from '15 years' to 15.
            df['age'] = df['age'].str.split(" y").str[0].astype(int)
            
            # Define age ranges
            age_bins = range(df['age'].min(), df['age'].max() + 11, 10)
            
            # Create labels for the age ranges
            labels = [f'{i}-{i+9}' if i != 100 else f'{i}+' for i in age_bins[:-1]]
            
            # Use pd.cut() to sort ages into age ranges
            df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=labels, right=False)
            
            # Applying the function to create the new column
            df['pop_proj_age'] = df.groupby(['municipality', 'age_group', 'year'])['pop_proj_all'].transform(sumval)
            
            return df
        
        except Exception as e:
            print(f"An error occurred during data processing: {e}")
            return None
        
    def sort_data(self, df):
        try:
            '''We clean up our dataframe further by sorting our columns, and dropping duplicate values for the age groups. 
            Now, we are able to drop the columns 'age' and 'prop_proj_all' as they are now obsolete.
            Finally, we reset the index of the dataframe.'''
            # Sorting the dataframe
            df = df.sort_values(by=['municipality', 'year', 'age'])
            
            # Ensuring the dataframe only has one observation for each age group
            df_unique = df.drop_duplicates(subset=['municipality', 'year', 'pop_proj_age'])
            
            # Dropping irrelevant columns
            columns_to_drop_2 = ['age', 'pop_proj_all']
            df = df_unique.drop(columns_to_drop_2, axis=1)
            
            # Resetting the index
            df.reset_index(inplace=True, drop=True)  # Drop old index too
            
            # Renaming the column
            df.rename(columns={'pop_proj_age': 'population_projection'}, inplace=True)
            
            return df
        
        except Exception as e:
            print(f"An error occurred during data sorting: {e}")
            return None
    
    def summarize_projection(self, df):
        try:
            '''To analyze the whole country, we add 'Denmark' as a value in the municipality column. We do it for every combination of year and age group.'''
            # Calculate the sum of population_projection for each combination of 'year' and 'age_group'
            summarized_data = df.groupby(['year', 'age_group'])['population_projection'].sum().reset_index()
            
            # Concatenate the original DataFrame with the summarized data
            df = pd.concat([df, summarized_data], ignore_index=True)
            
            # Determine the index where the new rows start
            original_length = len(df) - len(summarized_data)
            
            # Set the municipality for the newly added rows to 'Denmark'
            df.loc[original_length:, 'municipality'] = 'Denmark'
            
            return df
        
        except Exception as e:
            print(f"An error occurred during summarization: {e}")
            return None

    def calculate_total_population(self, df):
        try:
            '''To analyse on the total municipality for a given year, we make a new variable that sums up all the age groups.
            We group the data by municipality and year, and call these 'sum of ages' and add them to our dataframe.'''
            # Create a group by municipality and year
            summed_population = df.groupby(['municipality', 'year'])['population_projection'].sum().reset_index()
            total_population = summed_population.copy()
            
            # Call the total projected population 'sum of ages'
            total_population['age_group'] = 'sum of ages'
            
            # Calculate the sum for each group
            total_population['population_projection'] = summed_population.groupby(['municipality', 'year'])['population_projection'].transform('sum')
            
            # Concatenate the total population onto the existing dataframe
            df = pd.concat([df, total_population], ignore_index=True)
            
            # Sort the dataset for clarity
            df = df.sort_values(by=['municipality', 'year'])
            
            return df, total_population
        
        except Exception as e:
            print(f"An error occurred during total population calculation: {e}")
            return None, None

    def calculate_population_share(self, df, total_population):
        try:
            '''We want to look at the share of the population that each age group has in each municipality in each year. 
            We therefore create a new variable 'pop_share'.'''
            # Create a new column 'pop_share' to find the population share for each age group
            df = df.merge(total_population[['municipality', 'year', 'population_projection']], on=['municipality', 'year'], suffixes=('', '_total'))
            df['pop_share'] = df['population_projection'] / df['population_projection_total']
            
            # Drop the column 'population_projection_total' as we do not need it in our dataframe
            df = df.drop('population_projection_total', axis=1)
            
            return df
        
        except Exception as e:
            print(f"An error occurred during population share calculation: {e}")
            return None
        
    def generate_descriptive_table(self, df):
        # Get the number of unique municipalities, years, and age groups
        num_years = df['year'].nunique()
        num_age_group_2023 = df[(df['year'] == 2023) & (df['age_group'] != 'sum of ages')]['age_group'].nunique()
        num_municipalities = df[(df['year'] == 2023) & (df['municipality'] != 'Denmark')]['municipality'].nunique()

        # Create the min and max years:
        min_year = df['year'].min()
        max_year = df['year'].max()

        # Filter the DataFrame for the year 2023, age group 'sum of ages', and excluding the municipality 'Denmark'
        filtered_data = df[(df['year'] == 2023) & (df['age_group'] == 'sum of ages') & (df['municipality'] != 'Denmark')]

        # Calculate the descriptive statistics for the population projection
        projection_stats = filtered_data['population_projection'].describe()
        projection_median = filtered_data['population_projection'].median()

        descriptive_table = pd.DataFrame({
            'Attribute': ['Number of Municipalities', 'Number of Years', 'First Year', 'Last Year', 'Number of Age Groups',
                          'Min Population (2023)', 'Max Population (2023)',
                          'Mean Population (2023)', 'Median Population (2023)'],
            'Value': [num_municipalities, num_years, min_year, max_year, num_age_group_2023,
                      projection_stats['min'], projection_stats['max'],
                      projection_stats['mean'], projection_median]
        })

        # Convert scientific notation to normal numbers
        descriptive_table['Value'] = descriptive_table['Value'].apply(lambda x: '{:,.0f}'.format(x) if isinstance(x, (int, float)) else x)

        # Print the descriptive table
        print("Table 1: Descriptive Statistics")
        return descriptive_table
    
    def small_large_municipality(self, df):
        # Filter the DataFrame for the year 2023, age group 'sum of ages', and excluding the municipality 'Denmark'
        filtered_data = df[(df['year'] == 2023) & (df['age_group'] == 'sum of ages') & (df['municipality'] != 'Denmark')]

        # Calculate the descriptive statistics for the population projection
        projection_stats = filtered_data['population_projection'].describe()

        # Find the index of the row where the minimum population projection for the year 2023 occurs
        min_projection_index = filtered_data[filtered_data['population_projection'] == projection_stats['min']].index[0]
        
        # Retrieve the entire row corresponding to the minimum population projection
        min_projection_row = filtered_data.loc[min_projection_index]
        print("Municipality with the smallest population in 2023:")
        print(min_projection_row)
        
        # Find the index of the row where the maximum population projection for the year 2023 occurs
        max_projection_index = filtered_data[filtered_data['population_projection'] == projection_stats['max']].index[0]
        
        # Retrieve the entire row corresponding to the maximum population projection
        max_projection_row = filtered_data.loc[max_projection_index]
        print("Municipality with the largest population in 2023:")
        print(max_projection_row)
        
    def plot_population_projection(self, df):
        #We filter data, such that we only look at Denmark (not individual municipalities) and the sum of ages (not individual age groups)
        df_figure_1 = df[(df['municipality'] == 'Denmark') & (df['age_group'] == 'sum of ages')]

        # We plot the figure and format it appropriately
        plt.figure(figsize=(10, 6))
        plt.plot(df_figure_1['year'], df_figure_1['population_projection'], marker='', linestyle='-')
        plt.title('Figure 1: Population Projection for Denmark 2023 - 2050')
        plt.xlabel('Year')
        plt.ylabel('Population Projection (persons)')
        plt.grid(True)
        plt.ticklabel_format(style='plain')
        plt.show()

    def add_population_shares(self, df):
        #We make new columns in our DataFrame that we will use in the interactive figure. The columns consist of the population share by age group in 2023 and 2050.
        # Add pop_share_2023 and pop_share_2050 columns based on the year
        df['pop_share_2023'] = df.loc[df['year'] == 2023, 'pop_share']
        df['pop_share_2050'] = df.loc[df['year'] == 2050, 'pop_share']
        df['pop_share_2023'] = df['pop_share_2023'].fillna(0)
        df['pop_share_2050'] = df['pop_share_2050'].fillna(0)
        return df

    def plot_interactive(self, df, municipality): #The interactive figure
        # Filter the DataFrame for the selected municipality and the years 2023 and 2050
        filtered_df = df[(df['municipality'] == municipality) & 
                         (df['age_group'] != 'sum of ages') & 
                         (df['year'].isin([2023, 2050])) & 
                         (df['age_group'] != '100+')]  # Disregard the age group 100+ as the group is very small
        grouped_df = filtered_df.groupby('age_group')[['pop_share_2023', 'pop_share_2050']].sum().reset_index()

        # Plot the data
        ax = grouped_df.plot.bar(x='age_group', y=['pop_share_2023', 'pop_share_2050'], legend=True, width=0.4)
        ax.legend(['Population Shares in 2023', 'Population Shares in 2050'])
        ax.set_ylabel('Population Share')
        ax.set_xlabel('Age Group')
        ax.set_title(f'Figure 2: Population Share by Age Group for {municipality} in 2023 and 2050')
        plt.xticks(rotation=45)
        plt.show()