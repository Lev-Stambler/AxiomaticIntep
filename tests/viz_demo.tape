# VHS Demo for interventionfeatures viz command
# Run with: vhs tests/viz_demo.tape

Output tests/viz_demo.gif

Set FontSize 14
Set Width 800
Set Height 400
Set Theme "Dracula"

Type "interventionfeatures --help"
Enter
Sleep 2s

Type "# New viz command launches Streamlit visualization"
Enter
Sleep 1s

Type "interventionfeatures viz --help"
Enter
Sleep 2s
