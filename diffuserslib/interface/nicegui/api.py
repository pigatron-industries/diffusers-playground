from nicegui import ui, app

@app.get('/api/test')
def test():
    return {'test': 0}