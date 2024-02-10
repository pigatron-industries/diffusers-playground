from nicegui import app


@app.get('/api/test')
def test():
    return {'test': 0}