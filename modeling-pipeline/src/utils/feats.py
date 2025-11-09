import re

def select_columns(cols, dynamic_list, yearly_regex, static_regex):
    dyn = [c for c in dynamic_list if c in cols]
    yearly = [c for c in cols if yearly_regex and re.search(yearly_regex, c)]
    stat = [c for c in cols if static_regex and re.search(static_regex, c)]
    return dyn, yearly, stat
