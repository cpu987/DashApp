# add an index column 0  of HPI_master output to HPI_master2.csv
import pandas as pd
from datetime import datetime

# Read csv add date column from yr and period colums to dataframe
try:
    #df = pd.read_csv('data/HPI_master.csv') #adds just index
    # add date column from 'yr' and 'period' columns

    # custom date parser Ref: https://towardsdatascience.com/4-tricks-you-should-know-to-parse-date-columns-with-pandas-read-csv-27355bb2ad0e



    df = pd.read_csv('data/HPI_master.csv',
                     parse_dates={"date": ['yr', 'period']},
                     keep_date_col=True)


except FileNotFoundError:
    print("HPI_master.csv File not found.")
except pd.errors.EmptyDataError:
    print("No data in HPI_master.csv")
except pd.errors.ParserError:
    print("Parse error")
except Exception:
    print("Some other Read exception")

# write dataframe to csv include index column
try:
    df.to_csv(r'data/HPI_master_indexed.csv', index=True, header=True)

except FileNotFoundError:
    print("HPI_master2.csvFile not found.")
except pd.errors.EmptyDataError:
    print("No data in HPI_master2.csv")
except pd.errors.ParserError:
    print("Parse error")
except Exception:
    print("Some other Write exception")