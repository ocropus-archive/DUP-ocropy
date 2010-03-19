import matplotlib 
matplotlib.use('GTK') 

import gtk
def gtk_yield():
    while gtk.events_pending():
       gtk.main_iteration(False)
