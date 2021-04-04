import os
import pathlib
import unittest
from os import listdir
from os.path import join
from unittest.mock import patch

from scipy.io import wavfile

from main import prepare, register_user, get_users, login_user


class TestVoiceAuthentication(unittest.TestCase):
    files = ['data/x_authentication.npy', 'data/y_authentication.npy', 'data/x_identification.npy',
             'data/y_identification.npy', 'data/users.txt']

    def check_files_exist(self):
        for file in self.files:
            if not os.path.isfile(file):
                return False
        return True

    def test_prepare(self):
        for file in self.files:
            if os.path.isfile(file):
                os.remove(file)

        prepare()

        for file in self.files:
            self.assertTrue(pathlib.Path(file).is_file())

        users = get_users()
        self.assertTrue(len(users) > 0)

    @patch('main.record')
    def test_register_new_user(self, mock_record):
        prepare()

        users = get_users()
        first_users_len = len(users)
        new_user = 'liza'
        mock_record.return_value = wavfile.read('speakers/liza_register.wav')

        self.assertFalse(new_user in users)
        self.assertTrue(register_user(new_user))

        users = get_users()
        self.assertEqual(len(users), first_users_len + 1)
        self.assertTrue(new_user in users)

    def test_register_existed_user(self):
        if not self.check_files_exist():
            prepare()

        users = get_users()
        first_users_len = len(users)
        nickname = users[0]

        self.assertFalse(register_user(nickname))

        users = get_users()
        self.assertEqual(len(users), first_users_len)

    @patch('main.record')
    def test_login_legal_user_from_training_set(self, mock_record):
        if not self.check_files_exist():
            prepare()

        users = get_users()
        speaker_file = join(users[0], listdir(users[0])[0])
        mock_record.return_value = wavfile.read(speaker_file)
        self.assertTrue(login_user())

    @patch('main.record')
    def test_login_legal_user_not_from_training_set(self, mock_record):
        if not self.check_files_exist():
            prepare()

        users = get_users()
        speaker_file = join(users[0], listdir(users[0])[-1])
        mock_record.return_value = wavfile.read(speaker_file)
        self.assertTrue(login_user())

    @patch('main.record')
    def test_login_illegal_user_from_training_set(self, mock_record):
        if not self.check_files_exist():
            prepare()

        users = get_users()
        illegal_user = 'speakers\\russian\\male\\anonymous145'
        self.assertFalse(illegal_user in users)
        mock_record.return_value = wavfile.read(join(illegal_user, 'ru_0004.wav'))
        self.assertFalse(login_user())

    @patch('main.record')
    def test_login_illegal_user_not_from_training_set(self, mock_record):
        if not self.check_files_exist():
            prepare()

        users = get_users()
        illegal_user = 'speakers\\russian\\male\\anonymous145'
        self.assertFalse(illegal_user in users)
        mock_record.return_value = wavfile.read(join(illegal_user, 'ru_0014.wav'))
        self.assertFalse(login_user())

    @patch('main.record')
    def test_login_unknown_illegal_user(self, mock_record):
        if not self.check_files_exist():
            prepare()

        users = get_users()
        illegal_user = 'speakers\\russian\\male\\yalexand'
        self.assertFalse(illegal_user in users)
        mock_record.return_value = wavfile.read(join(illegal_user, 'ru_0042.wav'))
        self.assertFalse(login_user())


if __name__ == '__main__':
    unittest.main()
