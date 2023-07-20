import ipywidgets as widgets

INTERFACE_WIDTH = '900px'


def text(self, label, value) -> widgets.Text:
    text = widgets.Text(
        value=value,
        description=label,
        disabled=False,
        layout={'width': INTERFACE_WIDTH}
    )
    text.observe(self.onChange)
    return text


def intText(self, label, value) -> widgets.IntText:
    inttext = widgets.IntText(
        value=value,
        description=label,
        disabled=False
    )
    inttext.observe(self.onChange)
    return inttext


def floatText(self, label, value) -> widgets.FloatText:
    floattext = widgets.FloatText(
        value=value,
        description=label,
        disabled=False
    )
    floattext.observe(self.onChange)
    return floattext


def intSlider(self, label, value, min, max, step) -> widgets.IntSlider:
    slider = widgets.IntSlider(
        value=value,
        min=min,
        max=max,
        step=step,
        description=label,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        layout={'width': INTERFACE_WIDTH}
    )
    slider.observe(self.onChange)
    return slider


def floatSlider(self, label, value, min, max, step) -> widgets.FloatSlider:
    slider = widgets.FloatSlider(
        value=value,
        min=min,
        max=max,
        step=step,
        description=label,
        orientation='horizontal',
        readout=True,
        readout_format='.2f',
        layout={'width': INTERFACE_WIDTH}
    )
    slider.observe(self.onChange)
    return slider


def dropdown(self, label, options, value) -> widgets.Dropdown:
    dropdown = widgets.Dropdown(
        options=options,
        description=label,
        value=value,
        layout={'width': INTERFACE_WIDTH}
    )
    dropdown.observe(self.onChange)
    return dropdown

def textarea(self, label, value) -> widgets.Textarea:
    textarea = widgets.Textarea(
        value=value,
        description=label,
        layout={'width': INTERFACE_WIDTH, 'height': '100px'}
    )
    textarea.observe(self.onChange)
    return textarea


def checkbox(self, label, value) -> widgets.Checkbox:
    checkbox = widgets.Checkbox(
        value=value,
        description=label,
        disabled=False,
        indent=True
    )
    checkbox.observe(self.onChange)
    return checkbox