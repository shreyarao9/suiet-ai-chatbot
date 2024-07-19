from tkinter import *
from chat import get_response, bot_name

BG_GRAY = "#ABB2B9"
BG_COLOUR = "#17202A"
TEXT_COLOUR = "#EAECEE"

FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"

class ChatApp:

    def __init__(self):
        self.window = Tk()
        self._setup_main_window()

    def run(self):
        self.window.mainloop()

    def _setup_main_window(self):
        self.window.title("Chat (SUIET bot)")
        self.window.resizable(width=False, height=False) #makes window unresizable
        self.window.configure(width=470, height=550, bg=BG_COLOUR) #to give different attributes to widgets

        head_label = Label(self.window, bg=BG_COLOUR, fg=TEXT_COLOUR, text="Welcome", font=FONT_BOLD, pady=10) #label widget from tkinter
        head_label.place(relwidth=1)

        #divider
        line = Label(self.window, width=450, bg=BG_GRAY)
        line.place(relwidth=1, rely=0.07, relheight=0.012)

        #text widget stored in class variable since we're gonna use it in another function
        self.text_widget = Text(self.window, width=20, height=2, bg=BG_COLOUR, fg=TEXT_COLOUR, font=FONT, padx=5, pady=5)
        self.text_widget.place(relheight=0.745, relwidth=0.974, rely=0.08)
        self.text_widget.configure(cursor="arrow", state=DISABLED)

        #scrollbar = Scrollbar(self.text_widget)
        #scrollbar.place(relheight=1, relx=0.974)
        #scrollbar.configure(command=self.text_widget.yview) #changes y-pos of text widget whenever you scroll

        scrollbar = Scrollbar(self.window, command=self.text_widget.yview)
        scrollbar.place(relheight=0.745, relx=0.974, rely=0.08)
        self.text_widget.config(yscrollcommand=scrollbar.set)

        bottom_label = Label(self.window, bg=BG_GRAY, height=80)
        bottom_label.place(relwidth=1, rely=0.825)

        #message entry box
        self.message_entry = Entry(bottom_label, bg="#2C3E50", fg=TEXT_COLOUR, font=FONT)
        self.message_entry.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.011)
        self.message_entry.focus()
        self.message_entry.bind("<Return>", self._on_enter_pressed)

        send_button = Button(bottom_label, text="Send", font=FONT_BOLD, width=20, bg=BG_GRAY, command=lambda: self._on_enter_pressed(None))
        send_button.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)

    def _on_enter_pressed(self, event):
        msg = self.message_entry.get()
        self._insert_msg(msg, "You")

    def _insert_msg(self, msg, sender):
        if not msg:
            return 
        
        self.message_entry.delete(0, END)
        msg1 = f"{sender}: {msg}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg1)
        self.text_widget.configure(cursor="arrow", state=DISABLED)

        msg2 = f"{bot_name}: {get_response(msg)}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg2)
        self.text_widget.configure(cursor="arrow", state=DISABLED)

        self.text_widget.see(END)

if __name__ == "__main__":
    app = ChatApp()
    app.run()