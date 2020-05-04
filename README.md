![InvestiGame Logo](/images/logo.png)

# InvestiGame

Investors, resellers, and gaming enthusiasts are always interested in knowing if a new video game will be commercially sucessful. This information can be used in a variety of ways but most importantly as a signal for investment decisions since the gaming market is very volatile.

InvestiGame uses Machine Learning tools and the `AutoClassifier` package to calculate and predict how video games will perform on the market using social media text analytics, publisher/developer info, and professional reviews for the game. New games that are about to be released are searched for, the model retrained, and added to the database each day through `Cron`. This backend is then deployed into production as a WebApp using `Flask`, `GUnicorn`, and Amazon Web Services as well as an Android APK app developed using `Kivy` and `Buildozer`. The web app can be accessed at [**http://www.processwith.me**](http://www.processwith.me) and the Android app can be compiled using `Buildozer` from the source files in the KivyApp folder.

This project was developed as part of the [**Insight Data Science**](https://www.insightdatascience.com/) program at Toronto, Jan 2020 session. The end-to-end pipeline was developed in 4 weeks with the timeline being:
- 1st Week: brainstorming and feedback
- 2nd Week: EDA, model MVP
- 3rd Week: web-app MVP
- 4th Week: polished MVP and code, presentation for business case

### Contact
- Author: Ashwin Rai
- Email: ashwin2rai@gmail.com
- Linkedin: https://www.linkedin.com/in/ashwin2rai/
- Slides: https://bit.ly/investigame
- InvestiGame Website: http://processwith.me

### Usage

1. Install the `InvestiGame` package and the `AutoClassifier` package found in the project_workdir/ directory
2. Gain Reddit API (PRAW) access by registering for a developer account. Use the `utils` functions in the `InvestiGame` package to enter PRAW credentials. 
- To make use of SQL functions use the `InvestiGame utils` subpackage to create a SQL credentials dictionary which will be used to gain access to the PostgreSQL database.
3. Configure paths in the `config.py` file in both the `InvestiGame` and `AutoClassifier` package. 
4. Create a new environment, ideally using `conda`, and use the environment file in the env/ directory to install all required dependencies.
5. Run `project_workdir/script.py` in the new environment to execute InvestiGame. This will scrape data, preprocess, train a model and generate predictions. Finally the data can be saved as a csv file or pushed to a PostgreSQL database.

### The AutoClassifier Package

AutoClassifier is a stand alone package, developed by the author, that can be used to create a pipeline to perform binary classification almost instantly. It performs data cleaning, converts eligible features into categories, imputatation, feature selection, NLP, fits several classification models including a DNN model, performs hyperparameter tuning, and generates predictions with minimal user input.

### Directory navigation

- **data**: Folder used to dump data after package execution. Contains Genre map CSV and excel file showing stock variation with product release
- **env**: Folder containing `conda` environment file
- **images**: Folder containing images used in Readme
- **project_workdir**: Primary folder with source code
  - **autoclassifier**: Folder containing all source code for the `AutoClassifier` package
  - **investigame**: Folder containing all source code for the `InvestiGame` package
  - **FlaskApp**: Folder containing source code for Web App executed using `Flask`
    - **templates**: Folder containing HTML templates used by the Web App to render pages
    - *server.py*: Python source code for Web App
  - **KivyApp**: Folder containing source code for compiling Android App using `Buildozer` and `Kivy`
    - **font**: Folder containing fonts for Android App
    - **image**: Folder containing app image assets
    - **kv**: Folder containing user interface design instructions for `Kivy` screens
    - *main.kv*: KV source file for the primarry app user interface
    - *main.py*: Python source code for execution using `Kivy`
    - *buildozer.spec*: Specification file for compiling `Kivy` app into Android app using `Buildozer`
  - *requirements.txt*: Details dependencies for the packages
  - *setup.py*: Setup file for use with `pip`
  - *script.py*: Simple but complete execution of the `InvestiGame` package
  - *test_func.py*: Basic unit testing for `InvestiGame` package, use with `pytest`
- *LICENSE*: MIT license
- *.gitignore*: Files to ignore in Git
- *README.md*: This readme file
    
### Android App

The InvestiGame backend is also deployed as an Android App using `Kivy` and `Buildozer`. The InvestiGame package is used to generate a database of games and predictions which is uploaded into a `Google Firebase` database. The JSON data is pulled using the `Firebase` REST API by the Android app to display predictions of commercial performance for new, upcoming, and old games.  

![Screenshot of Android App Home Screen](/images/kivy.jpg)



