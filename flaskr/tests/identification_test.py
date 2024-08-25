import io

import pytest

from flaskr import create_app


@pytest.fixture()
def app():
    app = create_app({
        'TESTING': True,
    })

    # other setup can go here

    yield app

    # clean up / reset resources here


@pytest.fixture()
def client(app):
    return app.test_client()


def test_identify_mushroom_functional(client, mocker):
    mocker.patch('flaskr.identification.get_image_classification', return_value='species_name')

    response = client.post('/classifications/identify', data={
        'file': (io.BytesIO(b'file content'), 'test.jpg')
    })
    assert response.status_code == 200
    assert b'species_name' in response.data


def test_identify_mushroom_file_wrong_type(client):
    response = client.post('/classifications/identify', data={
        'file': (io.BytesIO(b'file content'), 'test.txt')
    })
    assert response.status_code == 400
    assert b'Invalid file type' in response.data


def test_identify_mushroom_missing_file(client):
    response = client.post('/classifications/identify', data={
    })
    assert response.status_code == 400
    assert b'No file found' in response.data

def test_identify_mushroom_no_file_selected(client, mocker):
    response = client.post('/classifications/identify', data={
        'file': (io.BytesIO(b''), '')
    })
    assert response.status_code == 400
    print(response.data)
    assert b'No selected file' in response.data
