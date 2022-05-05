# Viz-iT Backend

Django Rest API for a Visual Search Engine. The main concept of this visual search engine project is to use an image as
a search term. First segment that image into one or more objects, recognise those objects and then use the object
classification to get the object class. Then, use that object class as the search term query. The motivation for a
visual search engine is to provide the end-user vetted, trusted, and valuable content to their searchers based on the
given image as their search term query. This project will offer many applications such as: identifying objects within a
given image that users didn't know about, learning new objects and helping users find information on the internet in
terms of the image being the search query.

### Prerequisites

What things you need to install the software and how to install them

```
install python 3.8.0+
```

```
pip install pip
```

## Installation

Clone the repo

Create a virtual environment on your local machine and activate it

```
$ virtualenv venv
$ . venv/bin/activate
(venv) $ pip install -r requirements.txt
```

Install all the required pip modules

```bash
pip install -r requirements.txt
```

## Running the server

Running the server

```bash
python manage.py runserver
```

## Deployment

Open a web browser and paste the local-host http link. E.g. http://127.0.0.1:8000/

## Built With

* Django
* Python

## Contributing

* [Kay Thangan](https://github.com/KayThangan)

## Authors

* [Kay Thangan](https://github.com/KayThangan)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details