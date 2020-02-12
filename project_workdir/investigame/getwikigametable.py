from .utils import get_web_content

import pandas as pd

class GetWikiGameTable:
    
    def __init__(self):
        self.assembled_list = None
        self.game_df = None
        self.xpath_0_m = '/html/body/div[3]/div[3]/div[4]/div/table[2]//tr[{}]//text()'
        self.html_add_0_m = 'https://en.wikipedia.org/wiki/List_of_PlayStation_4_games'
        self.xpath_m_z = '/html/body/div[3]/div[3]/div[4]/div/table/tbody/tr[{}]//text()'
        self.html_add_m_z = 'https://en.wikipedia.org/wiki/List_of_PlayStation_4_games_(M-Z)'
    
    
    def get_wiki_table_list(self, addr, xpath, counter = 3):
        """
        Returns a list of list containing the first 7 values from each row of a Wikitable
    
        Parameters
        ----------
        addr: str
            webpage address
        xpath: str with placeholder
            the xpath address to pull data from. Requires a placeholder to specify a div tag
            example: '/head/div[{}].'
        counter: int
            the starting point of the xpath address. Used to skip the headers of a WikiTable
        assembled_list: list of lists, optional
            the output of get_wiki_table_list. Can be used to aggregate data from multiple tables in a recursive way.
            Default = None
        Returns
        -------
        List of lists
        A list of the first 7 values extracted from the row of a wiki table
        
        Notes
        -----
        This function is specifically used to pull values from rows of a specific WikiTable
        """
        tree = get_web_content(addr,'html')
        html_text = self._get_table_row(tree = tree, xpath = xpath, counter =counter)
    
        if not self.assembled_list:
            self.assembled_list = []
    
        while html_text:
            self.assembled_list.append(self._get_cleaned_row(html_text))
            counter += 1
            html_text = self._get_table_row(tree, xpath, counter)
        
        return self

    
    def _get_table_row(self, tree, xpath, counter):
        """
        Returns the output of html.xpath(xpath address) given an xpath address.
        
        Parameters
        ----------
        tree: lxml.html object  
            the object returned by the html function of the lxml module.
            Generally used with requests package. 
        xpath: str with placeholder
            the xpath address to pull data from. Requires a placeholder to specify a div tag
            example: '/head/div[{}].'
        counter: int
            specifies the tag in the xpath address
            
        Returns
        -------
        tree.xpath(xpath.format(counter))
        
        Raises
        ------
        Error (str): if tree.xpath(xpath.format(counter)) cannot return an object
            
        Notes
        -----
        This function is specifically used to pull rows from a WikiTable
        """
        try:
            return tree.xpath(xpath.format(counter))
        except:
            print('Error: Cannot extract table rows, check Xpath path and/or row index where content starts (int(counter))')
        

    def _get_cleaned_row(self, html_text):
        """
        Returns cleaned values from a row of a wiki table
        
        Parameters
        ----------
        html_text: str  
            text extracted from the html script of a row of a wiki table 
        Returns
        -------
        List
            A list of the first 7 values extracted from the row of a wiki table
            
        Notes
        -----
        This function is specifically used to pull values from rows of a specific WikiTable
        """    
        cleaned_html_text=[]
        
        for i in range(len(html_text)-1):
            if html_text[i] != '\n' and html_text[i+1] != '\n':
                j = 1
                while html_text[i+j] != '\n':
                    html_text[i] = html_text[i] + html_text[i+j]
                    html_text[i+j] = '\n'
                    j += 1
    
        for elem in html_text:
            if elem != '\n':
                cleaned_html_text.append(elem.replace('\n',''))
        
        return cleaned_html_text[0:7]
    
    def get_wiki_table_df(self, full_list = None,
                         columns=['Titles','Genres','Developers','Publishers','ReleaseDate_JP','ReleaseDate_EU','ReleaseDate_NA'],
                         release_list = ['ReleaseDate_JP','ReleaseDate_EU','ReleaseDate_NA']):
        """
        Returns a dataframe from a given list of lists, given 7 column data. 
        The last three columns are converted to datetime format.
        
        Parameters
        ----------
        full_list: List of lists  
        
        columns: List of str, optional
            This will be used to create the column headers of the dataframe.
            The last three columns 
            Default ['Titles','Genres','Developers','Publishers','ReleaseDate_JP','ReleaseDate_EU','ReleaseDate_NA']
            
        Returns
        -------
        Pandas.DataFrame object
        
        Raises
        ------
        Error (str): if issues with list of list commonly from scraping issues
        Error (str): if issues with datetime conversion
        
        Notes
        -----
        This function is specifically used to convert a specific WikiTable scraped data into a dataframe.
        """    
        
        if not full_list:
            full_list = self.assembled_list
        
        try:
            ps4_game_list = pd.DataFrame(full_list, columns = columns)
        except:
            print('Error: issues with converting scraped wiki table data into DataFrame.')
            return None
    
        for col in release_list:
            try:
                ps4_game_list[col]=pd.to_datetime(ps4_game_list[col],infer_datetime_format=True,errors='coerce')
            except:
                print('Could not convert scraped release date columns into datetime format.')
    
        punc_dict={ord('\''):None, ord(':'):None, ord('#'):None, ord('/'):' ', 
               ord('&'):None, ord(';'):' ', ord('!'):None, ord(','):None, ord('?'):None, ord('.'):None}        
        
        ps4_game_list['TitlesHTML'] = ps4_game_list['Titles'].str.replace('\.0',' 0').str.replace('\.1',' 1').str.replace('\.5',' 5').str.replace(' -',' ').str.replace('//',' ').str.translate(punc_dict).str.replace('  ',' ').str.strip().str.lower().str.replace(' ','+')
        
        self.game_df = ps4_game_list
        