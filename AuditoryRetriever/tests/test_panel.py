import panel as pn
import pandas as pd
import numpy as np
import hvplot.pandas

# Step 1: Initialize Panel extension
pn.extension()

# Step 2: Create a sample DataFrame for the static plot
static_data = pd.DataFrame({
    'x': np.linspace(0, 10, 100),
    'y': np.sin(np.linspace(0, 10, 100))
})

# Step 3: Create a simple static plot using hvplot
static_plot = static_data.hvplot.line(x='x', y='y', title='Static Sine Wave')

# Step 4: Define a function to change button color
def toggle_button_color(event):
    if button.button_style == 'solid':
        button.button_style = 'outline'
    else:
        button.button_style = 'solid'

# Step 5: Create a reactive button for the static plot
button = pn.widgets.Button(name='Toggle Color', button_type='primary', button_style='solid')
button.on_click(toggle_button_color)

# Step 6: Create a Panel layout to display the static plot and button
static_layout = pn.Column(
    pn.pane.Markdown("# Static Sine Wave Example"),
    static_plot,
    button
)

# Sinus function for the updating plot
def sinus_function(x):
    return np.sin(x)

# Prepare data for the updating plot
x_data = np.linspace(0, 10, 100)
y_data = sinus_function(x_data)

# Create a DataFrame for the updating plot
updating_df = pd.DataFrame({'x': x_data, 'y': y_data})

# Function to generate updating hvplot
def hvplot_update():
    for i in range(len(updating_df)):
        yield updating_df.iloc[:i+1].hvplot.line(x='x', y='y', ylim=(-1, 1))

# Panel to hold the updating plot
hvplot_panel = pn.Column()

# Function to update the hvplot
def update_hvplot():
    for plot in hvplot_update():
        hvplot_panel.objects = [plot]
        pn.io.push_notebook()

# Create a final Panel layout to display all components
final_layout = pn.Row(
    static_layout,
    pn.Column(
        pn.pane.Markdown("# Updating Sine Wave Example"),
        hvplot_panel
    )
)

# Run the update function in a separate thread to simulate the animation
import threading

def run_update():
    update_hvplot()

thread = threading.Thread(target=run_update)
thread.start()

# Step 7: Serve the Panel application
pn.serve(final_layout)
