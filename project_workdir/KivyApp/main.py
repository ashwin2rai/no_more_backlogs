from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen
from kivy.uix.image import AsyncImage

import certifi
import os
import webbrowser
import requests


class HomeScreen(Screen):
    pass


class GameScreen(Screen):
    pass


class NoIntScreen(Screen):
    pass


class InfoScreen(Screen):
    pass


class ResultScreen(Screen):
    pass


class MainApp(App):

    # Here's all the magic !
    os.environ['SSL_CERT_FILE'] = certifi.where()

    spin_val = 'Default'
    firebase_url = 'https://APPID.firebaseio.com/'
    game_card_url = 'http://store.playstation.com/store/api/chihiro/00_09_000/container/US/en/19/UP4101-CUSA00856_00-SOCIALSHAREARENA/image?w=320&h=320&bg_color=000000&opacity=100&_version=00_09_000'

    try:
        data_req_titles = requests.get(firebase_url+'Titles.json').json()
        data_req_bool = True
    except:
        data_req_titles = ['1','2','3']
        data_req_bool = False

    def on_start(self):
        #get database data
        self.game_card_im = AsyncImage(source=self.game_card_url)
        self.root.ids['result_screen'].ids['game_card_layout'].add_widget(self.game_card_im)


    def change_screen(self, screen_name, step=False):
        if step:
            if not self.data_req_bool:
                self.change_screen('noint_screen')
            else:
                self.change_screen(screen_name)
        else:
            screen_manager = self.root.ids['screen_manager']
            screen_manager.current = screen_name

    def open_webpage(self, url):
        webbrowser.open(url)

    def save_spin_val(self, text):
        self.spin_val = f'{self.data_req_titles.index(text)}.json'
        rest_url = lambda x: requests.get(f'{self.firebase_url}{x}/'+self.spin_val).content.decode()
        self.game_card_url =  rest_url('GameCard')[1:-1].replace('https','http',1)
        self.root.ids['result_screen'].ids['game_card_layout'].remove_widget(self.game_card_im)
        self.game_card_im = AsyncImage(source=self.game_card_url)
        self.root.ids['result_screen'].ids['game_card_layout'].add_widget(self.game_card_im)
        self.root.ids['result_screen'].ids['game_det_text'].text = f"Game: {text}\nPublisher: {rest_url('Publishers')[1:-1]}\nDeveloper: {rest_url('Developers')[1:-1]}\nAggregate Review: {rest_url('Reviews')}"
        self.root.ids['result_screen'].ids['game_pred_text'].text = f"Prediction: This game will be a {rest_url('SuccessPredictText')[1:-1]}\nConfidence: {float(rest_url('SuccessPredictProb'))*100:.2f}%"

    def build(self):
        return Builder.load_file('main.kv')

MainApp().run()
