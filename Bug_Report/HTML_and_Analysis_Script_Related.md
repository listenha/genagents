# Bug Report: HTML and Analysis Script Related Issues

This document catalogs bugs encountered during the development and maintenance of the `Surveys/analyze_consistency.py` script and its generated interactive HTML reports. Each entry includes the bug description, error message (if applicable), root cause, solution, and prevention tips.

---

## 1. IndentationError: Unexpected Indent

**Error Message:**
```
IndentationError: unexpected indent (Surveys/analyze_consistency.py:969)
```

**Description:**
The HTML template string was prematurely closed, causing subsequent HTML code to be interpreted as Python code, resulting in an indentation error.

**Root Cause:**
The f-string containing the HTML template had a closing `"""` in the wrong place, ending the string before all HTML content was included.

**Solution:**
Removed the premature closing `"""` to extend the f-string to include the complete HTML template, including the trajectory section.

**Prevention:**
- Always verify that multi-line f-strings are properly closed at the end of the intended content
- Use an IDE with syntax highlighting to visually identify string boundaries
- Check that all HTML sections are within the f-string before the closing `"""`

---

## 2. HTML File Too Large (204MB) and Graphs Not Rendering

**Error Message:**
- Browser takes >5 minutes to load local HTML file
- Graphs appear blank/invisible in the "Question Level Consistency" tab

**Description:**
The generated HTML file was extremely large (204MB) and graphs were not rendering, making the report unusable.

**Root Cause:**
**Combinatorial Explosion**: All Plotly chart data for all views (raw/distilled × all/question-agg/agent-agg) was being generated upfront and embedded inline as full JSON within the HTML. This created an exponential growth in file size:
- For 200 agents and multiple views, this resulted in massive JSON data embedded in the HTML
- The "compare two agents" feature was generating charts for all agent pairs, further exacerbating the problem

**Solution:**
Refactored the `generate_html_report` function to use **dynamic client-side chart generation**:
1. **Default view only**: Generate full HTML (with embedded Plotly scripts) only for the default view ("Raw Responses" → "All" heatmap)
2. **Lightweight data storage**: For all other views, store only the underlying DataFrame data (e.g., `question_id`, `icc_2k`) as lightweight JSON
3. **Dynamic chart generation**: Implemented JavaScript functions (`generateBarChart`, `generateHeatmap`, `loadChartsForView`) to dynamically generate and render Plotly charts client-side from the lightweight JSON data when users switch views
4. **Removed comparison feature**: Removed the "compare two agents" feature that was generating charts for all agent pairs

**Code Changes:**
- Created `chart_data_js` variable containing lightweight JSON data and JavaScript functions for dynamic chart generation
- Modified `updateQuestionLevelView()` to trigger `loadChartsForView()` for non-default views
- Removed the loop generating comparison charts for all agent pairs

**Prevention:**
- **Avoid embedding large data structures**: Never embed full chart JSON for all possible views in HTML
- **Use lazy loading**: Generate charts only when needed (when user switches views)
- **Consider data size**: Before embedding data, estimate the total size (number of views × data points × JSON overhead)
- **Monitor file size**: If HTML file exceeds 10-20MB, investigate and optimize

---

## 3. Uncaught SyntaxError: F-string Escaping Issues

**Error Messages:**
```
Uncaught SyntaxError: f-string: single '}' is not allowed (Surveys/analyze_consistency.py:1943)
Uncaught SyntaxError: Unexpected token '}' (consistency_report_0000-0199.html:387)
```

**Description:**
JavaScript code embedded within Python f-strings contained single `}` characters that were not properly escaped, causing syntax errors.

**Root Cause:**
When JavaScript code is embedded in a Python f-string, single `{` and `}` characters are interpreted as f-string syntax. To include literal braces in the output, they must be escaped as `{{` and `}}`. However, this was inconsistently applied:
- Some JavaScript code used `{{` and `}}` correctly
- Other parts used single `{` and `}` incorrectly
- The `js_code` variable is a **regular string** (not an f-string), so it should use single braces `{` and `}`
- The `chart_data_js` variable is an **f-string**, so it correctly uses double braces `{{` and `}}`

**Solution:**
1. **For `js_code` (regular string)**: Use single braces `{` and `}` - these are literal JavaScript braces
2. **For `chart_data_js` (f-string)**: Use double braces `{{` and `}}` to escape them in the f-string
3. Fixed all functions in `js_code`:
   - `updateTrajectoryView()`: Changed `}}` to `}`
   - `updateTraitSelectorVisibility()`: Changed `{{` to `{` and `}}` to `}`
   - `updateTrajectoryInstrumentFilter()`: Changed all `{{` to `{` and `}}` to `}`

**Code Example:**
```python
# CORRECT: js_code is a regular string, use single braces
js_code = """
    function updateTrajectoryView() {
        // JavaScript code with single braces
        if (condition) {
            doSomething();
        }
    }
"""

# CORRECT: chart_data_js is an f-string, use double braces
chart_data_js = f"""
    function generateBarChart() {{
        var data = {{
            x: [1, 2, 3],
            y: [4, 5, 6]
        }};
    }}
"""
```

**Prevention:**
- **Identify string type**: Determine if the string is a regular string or an f-string
- **Regular strings**: Use single braces `{` and `}` for JavaScript code
- **F-strings**: Use double braces `{{` and `}}` to escape literal braces
- **Consistency check**: When copying JavaScript code between regular strings and f-strings, remember to adjust brace escaping
- **Test syntax**: Always validate JavaScript syntax after embedding in Python strings

---

## 4. TypeError: Unhashable Type 'dict'

**Error Message:**
```
TypeError: unhashable type: 'dict' (Surveys/analyze_consistency.py:1755)
```

**Description:**
A Python dictionary literal was incorrectly using f-string escaping syntax (`{{` and `}}`) instead of regular braces.

**Root Cause:**
The code was inside an f-string context, but the dictionary literal was Python code (not JavaScript), so it should use regular braces `{` and `}`, not escaped braces `{{` and `}}`.

**Solution:**
Changed `{{` and `}}` to `{` and `}` in the Python dictionary literal for `distilled_heatmap_data`.

**Prevention:**
- **Distinguish Python vs JavaScript**: Python code within f-strings uses regular braces; only JavaScript/HTML that needs literal braces in output requires escaping
- **Python dictionaries**: Always use `{` and `}` in Python code, even within f-strings

---

## 5. Uncaught ReferenceError: showTab is not defined

**Error Message:**
```
Uncaught ReferenceError: showTab is not defined
    at HTMLButtonElement.onclick (consistency_report_0000-0199.html:529:74)
```

**Description:**
The `showTab` function was not available when the HTML buttons tried to call it, causing the tab switching functionality to fail.

**Root Cause:**
The JavaScript code containing `showTab` was not being correctly inserted into the HTML. The previous approach of replacing `</script>` tags was breaking the HTML structure, and the scripts were executing before the DOM was ready.

**Solution:**
1. **Manual placeholder replacement**: Modified the script insertion logic to manually replace the `{js_code}` placeholder in the HTML template with the actual `js_code` string
2. **Separate script blocks**: Inserted `chart_data_js` in a separate `<script>` block immediately after the Plotly CDN script tag, ensuring the CDN script is properly closed and loaded first
3. **DOMContentLoaded wrapper**: Wrapped Plotly-generated scripts in `DOMContentLoaded` event listeners to ensure they execute after the DOM is ready

**Code Changes:**
```python
# Replace js_code placeholder
html_content = html_content.replace('{js_code}', js_code)

# Insert chart data JavaScript in separate script block after Plotly CDN
html_content = html_content.replace(
    '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>',
    '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>\n    ' + all_scripts_html,
    1
)
```

**Prevention:**
- **Validate function existence**: After generating HTML, verify that key JavaScript functions are present
- **Check script insertion**: Ensure scripts are inserted in the correct location and order
- **DOM readiness**: Use `DOMContentLoaded` for scripts that manipulate DOM elements
- **Avoid nested script tags**: Extract JavaScript content from Plotly's `to_html()` output and wrap it properly

---

## 6. Graphs Not Appearing in "Question Level Consistency" Tab

**Error Message:**
- No visible error, but graphs were blank/invisible

**Description:**
Heatmaps in the "Question Level Consistency" tab under the "All" view for raw responses were not rendering, even though the HTML structure appeared correct.

**Root Cause:**
**Indentation error**: The `fig_heatmap` creation code was outside the `for` loop that iterated over agent chunks. This meant:
- Only one heatmap was being generated (or none if the loop didn't run correctly)
- The heatmap generation logic wasn't executing for all chunks

**Solution:**
Corrected the indentation of the `fig_heatmap` creation code to be inside the `for` loop, ensuring heatmaps are generated for all agent chunks.

**Code Location:**
- Lines 1129-1154: Heatmap generation loop indentation corrected

**Prevention:**
- **Verify loop scope**: Always check that code that should run inside a loop is properly indented
- **Test with small datasets**: Use a small number of agents (e.g., 10) to verify all expected charts are generated
- **Inspect generated HTML**: Check that the expected number of chart divs and scripts are present

---

## 7. NameError: script_pattern is not defined

**Error Message:**
```
NameError: name 'script_pattern' is not defined (Surveys/analyze_consistency.py:1591)
NameError: name 'html_str' is not defined (Surveys/analyze_consistency.py:1644)
```

**Description:**
The `extract_scripts` function had incorrect indentation, causing `script_pattern` and the `return` statement to be outside the function scope.

**Root Cause:**
**Indentation error**: The `script_pattern` variable and `return` statement were not properly indented within the `extract_scripts` function, causing them to be interpreted as module-level code.

**Solution:**
Corrected the indentation of `script_pattern` and the `return` statement to be inside the `extract_scripts` function.

**Prevention:**
- **Consistent indentation**: Use an IDE with automatic indentation or a linter to catch indentation errors
- **Function scope**: Always verify that variables and return statements are inside the function they belong to
- **Run syntax checks**: Use `ast.parse()` or similar tools to validate Python syntax before running

---

## 8. JSX Element 'script' Has No Corresponding Closing Tag

**Error Message:**
```
JSX element 'script' has no corresponding closing tag (consistency_report_0000-0199.html:7-9)
```

**Description:**
The HTML had nested `<script>` tags, which is invalid HTML and can cause parsing errors.

**Root Cause:**
1. **Nested script tags**: Plotly's `to_html()` returns full `<script>` tags, and these were being wrapped in another `<script>` tag, creating nested scripts
2. **DOM timing**: Scripts were placed in the `<head>` and executed before the DOM elements existed, causing `document.getElementById()` to return `null`

**Solution:**
1. **Extract JavaScript content**: Modified script insertion to extract only the JavaScript content (strip the `<script>` and `</script>` tags) from Plotly's output
2. **DOMContentLoaded wrapper**: Wrapped the extracted content in a `DOMContentLoaded` event listener
3. **Single script tag**: Placed the wrapped content within a single `<script>` tag, preventing nested scripts

**Code Changes:**
```python
# Extract JavaScript content from <script>...</script> tag
script_content = re.sub(r'^<script[^>]*>', '', script, flags=re.IGNORECASE | re.DOTALL)
script_content = re.sub(r'</script>\s*$', '', script_content, flags=re.IGNORECASE | re.DOTALL)

# Wrap in DOMContentLoaded
wrapped_script = f'''<script type="text/javascript">
    document.addEventListener('DOMContentLoaded', function() {{
        {script_content}
    }});
</script>'''
```

**Prevention:**
- **Avoid nested script tags**: Never wrap `<script>` content in another `<script>` tag
- **Extract content**: When using libraries that generate full HTML/script tags, extract only the content you need
- **DOM readiness**: Always use `DOMContentLoaded` for scripts that access DOM elements
- **Validate HTML**: Use HTML validators to catch structural issues

---

## 9. "1" and Grey Line Instead of Agent Range Label

**Error Message:**
- No error, but incorrect display: titles showed "1" followed by a light grey separation line instead of agent index range labels

**Description:**
In the "aggregated by agent" view (raw and distilled) and the "All" view for distilled scores, the agent index range label was missing. Instead, a "1" followed by a light grey separation line was displayed.

**Root Cause:**
The title format in the JavaScript `generateBarChart` and `generateHeatmap` functions was incorrect. The title was being set as `title: {text: title}` (an object), which Plotly may have rendered incorrectly, instead of `title: title` (a plain string).

**Solution:**
Changed the title format in both `generateBarChart` and `generateHeatmap` functions from:
```javascript
title: {text: title}  // INCORRECT
```
to:
```javascript
title: title  // CORRECT
```
This matches the format used in the working raw heatmaps.

**Code Location:**
- `generateBarChart` function (line ~1877): Changed `title: {text: title}` to `title: title`
- `generateHeatmap` function (line ~1903): Changed `title: {text: title}` to `title: title`

**Prevention:**
- **Consistent format**: Use the same title format across all chart generation functions
- **Match working examples**: When adding new chart types, use the same format as existing working charts
- **Test all views**: Verify that titles display correctly in all views (raw/distilled, all/question-agg/agent-agg)

---

## 10. Blank BES-A and REI Scores in Trajectory Graph

**Error Message:**
- No error, but BES-A and REI score trajectories were blank/not plotted

**Description:**
After modifying the script to use `normalized_score` for BES-A and REI (changing from `CE_total`/`RA_total` to `CE_normalized`/`RA_normalized`), the trajectory graphs showed blank lines for these scores.

**Root Cause:**
**Mismatch between data extraction and trajectory generation**: 
- The data extraction code was updated to use `'CE_normalized'` and `'RA_normalized'`
- However, the trajectory generation code and JavaScript filtering logic were still looking for `'CE_total'` and `'RA_total'`
- This mismatch caused the scores to not be found/plotted

**Solution:**
Updated all references to use the new normalized score names consistently:
1. **Python trajectory generation**:
   - `all_score_names`: Updated from `['CE_total', 'RA_total']` to `['CE_normalized', 'RA_normalized']`
   - `besa_scores`: Updated from `['CE_total']` to `['CE_normalized']`
   - `rei_scores`: Updated from `['RA_total']` to `['RA_normalized']`
   - `score_colors` dictionary: Updated keys from `'CE_total'` and `'RA_total'` to `'CE_normalized'` and `'RA_normalized'`
2. **JavaScript filtering**:
   - `updateTrajectoryInstrumentFilter()`: Updated `traceName.includes` checks from `'CE_total'` and `'RA_total'` to `'CE_normalized'` and `'RA_normalized'`
3. **Y-axis ranges**: Updated y-axis ranges for 'besa' and 'rei' from `[0, 50]` to `[0, 5]` to match the normalized score range

**Code Locations:**
- Lines 2106, 2110, 2111: Score name lists
- Lines 2102-2103: `score_colors` dictionary keys
- Lines 771-775: JavaScript `updateTrajectoryInstrumentFilter` function
- Lines 739-744: JavaScript `yAxisRanges` object
- Lines 2183, 2242: Python `fig_all_combined` and `fig_agent` y-axis ranges

**Prevention:**
- **Consistent naming**: When renaming variables/keys, update ALL references across the entire codebase
- **Search and replace**: Use IDE search-and-replace to find all occurrences
- **Verify data flow**: Trace data from extraction → processing → visualization to ensure names match
- **Test after changes**: Always test that all expected data is displayed after renaming

---

## 11. Incorrect Y-Axis Range for Trajectory Graph

**Error Message:**
- No error, but y-axis range was incorrect (0-50 instead of 0-5)

**Description:**
After normalizing BES-A and REI scores to the 1-5 range, the trajectory graph's y-axis still showed 0-50 for the "All Scores" filter, compressing the scores at the bottom.

**Root Cause:**
The y-axis range configuration was not updated to reflect the new normalized score range (1-5) for BES-A and REI.

**Solution:**
Updated y-axis ranges in both Python and JavaScript:
1. **JavaScript `yAxisRanges` object**: Updated `min` and `max` values for 'all', 'besa', and 'rei' from `[0, 50]` to `[0, 5]`
2. **Python figure generation**: Updated `fig_all_combined` and `fig_agent` y-axis ranges from `[0, 50]` to `[0, 5]`

**Prevention:**
- **Update all range references**: When changing data ranges, update all y-axis configurations (both Python and JavaScript)
- **Consistent ranges**: Ensure y-axis ranges match the actual data range
- **Test all filters**: Verify y-axis ranges are correct for all instrument filters

---

## General Prevention Strategies

### 1. **String Type Awareness**
- Always identify whether you're working with a regular string or an f-string
- Regular strings: Use single braces `{` and `}` for JavaScript/Python code
- F-strings: Use double braces `{{` and `}}` to escape literal braces in output

### 2. **Indentation Validation**
- Use IDE with automatic indentation
- Run syntax checks (`ast.parse()`) before executing
- Verify loop and function scope

### 3. **File Size Monitoring**
- Monitor generated HTML file size
- If file exceeds 10-20MB, investigate and optimize
- Use lazy loading and dynamic generation for large datasets

### 4. **Data Consistency**
- When renaming variables/keys, update ALL references
- Use search-and-replace to find all occurrences
- Trace data flow from extraction to visualization

### 5. **HTML Structure Validation**
- Avoid nested script tags
- Use `DOMContentLoaded` for DOM manipulation
- Validate HTML structure with validators

### 6. **Testing Checklist**
- Test with small datasets first
- Verify all views (raw/distilled, all/question-agg/agent-agg)
- Check that all expected charts are generated
- Verify JavaScript functions are defined and accessible
- Test all filter combinations

### 7. **Code Organization**
- Keep JavaScript code in separate variables (`js_code`, `chart_data_js`)
- Use clear naming conventions
- Document which strings are f-strings vs regular strings

---

## Summary of Key Lessons

1. **F-string escaping is critical**: Always be aware of whether you're in an f-string context and escape braces accordingly
2. **Indentation errors are subtle**: Always validate indentation, especially in loops and functions
3. **Combinatorial explosion is real**: Avoid generating all possible views upfront; use lazy loading
4. **Data consistency matters**: When renaming, update ALL references across the entire codebase
5. **DOM timing is important**: Use `DOMContentLoaded` for scripts that access DOM elements
6. **File size matters**: Monitor HTML file size and optimize when it grows too large
7. **Test incrementally**: Test with small datasets and verify each feature before scaling up

---

## Notes for Future Maintenance

- When adding new features, refer to this document to avoid common pitfalls
- If new bugs are encountered, add them to this document with the same format (description, error, root cause, solution, prevention)
- Keep this document updated as the codebase evolves

