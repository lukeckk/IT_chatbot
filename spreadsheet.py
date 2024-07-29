import gspread
from oauth2client.service_account import ServiceAccountCredentials

scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
credentials = ServiceAccountCredentials.from_json_keyfile_name("ss_credentials.json", scope)
client = gspread.authorize(credentials) # client email

# sheet = client.create('IT Helper Test') # naming the sheet
# sheet.share('lukeecheng@gmail.com', perm_type='user', role='writer') # get sheet to appear on personal email google drive and insert 'writer' for ability to view and edit

# open the sheet
sheet = client.open("IT Helper Test").sheet1

header = sheet.row_values(1)    # get all values in the first row
# issue_column = sheet.acell('A1').value-
issue_column = header.index("Issue") + 1
issue_values = sheet.col_values(issue_column)
new_issue = 'issue one'
sheet.update_cell(len(issue_values) + 1, issue_column, new_issue)