import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.filedialog import askopenfile, askdirectory


## Creating main window
win = tk.Tk()
win.title('Ship Detection User Interface')
win.maxsize(800,500)
win.iconphoto(False,tk.PhotoImage(file='iirs.png'))
#win.geometry("750x500")
#win.configure(background = 'white')

# img = ImageTk.PhotoImage(Image.open('iirs-isro.jpg'))
# panel = tk.Label(win, image=img)
# panel.grid(row=0,column=0, sticky="NSEW")


#Functions

def choose_file():
    file = askopenfile(mode ='r', filetypes =[('TIFF files', '*.tif')])
    choose_file_var = file.name
    temp_name = choose_file_var.split('/')
    choose_btn_entry.configure(text='/'+str(temp_name[len(temp_name)-1]))

def choose_vfile():
    file = askopenfile(mode ='r', filetypes =[('Shape Files', '*.shp')])
    choose_vlayer_var = file.name
    temp_name = choose_vlayer_var.split('/')
    choose_btn_entry_vlayer.configure(text='/'+str(temp_name[len(temp_name)-1]))

def choose_folder():
    file = askdirectory()
    choose_dir_var = file
    temp_name = choose_dir_var.split('/')
    choose_btn_entry_dir.configure(text='/'+str(temp_name[len(temp_name)-1]))

def ismasked():
    if masksed_var.get():
        choose_btn_vlayer.configure(state='disabled')
        choose_btn_entry_vlayer.configure(text="*No need to Select Vector Shape file*")
    else:
        choose_btn_vlayer.configure(state='enabled')
        choose_btn_entry_vlayer.configure(text="")

def startDetection():
    inputfile = choose_file_var
    outputdir = choose_dir_var
    masked = masksed_var.get()
    print(inputfile)
    tw = tar_win_var.get()
    gw = guard_win_var.get()
    bw = background_win_var.get()
    pfa = pfa_win_var.get()
    if masked == 0:
        vector_layer = choose_vlayer_var
        if inputfile=="" or outputdir=="" or tw==0 or gw==0 or bw==0 or pfa==0 :
            messagebox.showerror("Value Error","Please Enter all the values!") 


#Heading
head_label = ttk.Label(win, text="Automatic Ship Detection Graphical User Interface\n")
head_label.configure(font=("Times New Roman",20,"bold"))
head_label.grid(row=0, columnspan=5,sticky=tk.NW)

## Labels
#inputfile
choose_file_label = ttk.Label(win, text="Select Input File: \n") 
choose_file_label.configure(font=("Times New Roman", 12))
choose_file_label.grid(row=1,column=1, sticky=tk.W)

## Is Masked
masked_label = ttk.Label(win,text="Is Input File Masked already?\n")
masked_label.configure(font=("Times New Roman", 12))
masked_label.grid(row=2,column=1, sticky=tk.W)

## Select outputDir
chooose_dir_label = ttk.Label(win, text="Select the Output Folder: \n")
chooose_dir_label.configure(font=("Times New Roman", 12))
chooose_dir_label.grid(row=3, column=1, sticky=tk.W)

##Select Vector Layer
chooose_vlayer_label = ttk.Label(win, text="Select the Vector file: \n")
chooose_vlayer_label.configure(font=("Times New Roman", 12))
chooose_vlayer_label.grid(row=4, column=1, sticky=tk.W)

#choose algorithm
algolabel = ttk.Label(win, text="Choose Algorithm: \n")
algolabel.configure(font=("Times New Roman",12))
algolabel.grid(row=5,column=1, sticky=tk.W)

# Choose target window size
tar_win = ttk.Label(win,text="Enter Target Window Size: \n")
tar_win.configure(font=("Times New Roman",12))
tar_win.grid(row=6,column=1,sticky=tk.W)

# Choose guard window size
guard_win = ttk.Label(win,text="Enter Guard Window Size: \n")
guard_win.configure(font=("Times New Roman",12))
guard_win.grid(row=7,column=1,sticky=tk.W)

# Choose background window size
Back_win = ttk.Label(win,text="Enter Background Window Size: \n")
Back_win.configure(font=("Times New Roman",12))
Back_win.grid(row=8,column=1,sticky=tk.W)

#choose channel
channellabel = ttk.Label(win, text="Choose Channel: \n")
channellabel.configure(font=("Times New Roman",12))
channellabel.grid(row=9,column=1, sticky=tk.W)

# Choose pfa
pfa_win = ttk.Label(win,text="Enter Pfa: \n")
pfa_win.configure(font=("Times New Roman",12))
pfa_win.grid(row=10,column=1,sticky=tk.W)


#Wigets########################################################################################################
#for input file
choose_file_var = ""
choose_btn = ttk.Button(win, text='Choose File\t', command=lambda:choose_file(),width=12)
choose_btn.grid(row=1,column=3,sticky=tk.N)


choose_btn_entry = ttk.Label(win, text="",width=80)
choose_btn_entry.configure(font=("Times New Roman",12))
choose_btn_entry.grid(row=1,column=4,sticky=tk.N)

##Radio button for isMaksed?
masksed_var = tk.IntVar()
masked_radio_yes = ttk.Radiobutton(win, text='Yes', variable=masksed_var, value=1, command=lambda:ismasked())
masked_radio_yes.grid(row=2,column=3,sticky=tk.N)

masked_radio_no = ttk.Radiobutton(win, text='No', variable=masksed_var, value=0, command=lambda:ismasked())
masked_radio_no.grid(row=2,columnspan=5,sticky=tk.N)


#for output directory
choose_dir_var = ""
choose_btn_dir = ttk.Button(win, text='Choose Folder\t', command=lambda:choose_folder(),width=12)
choose_btn_dir.grid(row=3,column=3,sticky=tk.N)


choose_btn_entry_dir = ttk.Label(win, text="",width=80)
choose_btn_entry_dir.configure(font=("Times New Roman",12))
choose_btn_entry_dir.grid(row=3,column=4,sticky=tk.N)


## Select Vector layer
choose_vlayer_var = ""
choose_btn_vlayer = ttk.Button(win, text='Choose File\t', command=lambda:choose_vfile(),width=12)
choose_btn_vlayer.grid(row=4,column=3,sticky=tk.N)


choose_btn_entry_vlayer = ttk.Label(win, text="",width=80)
choose_btn_entry_vlayer.configure(font=("Times New Roman",12))
choose_btn_entry_vlayer.grid(row=4,column=4,sticky=tk.N)


## Choose Algorithm
algo_var = tk.StringVar()
algo_combobox = ttk.Combobox(win, width=15, textvariable=algo_var, state='readonly')
algo_combobox['values'] = ('Select Algorithm','Standard_CFAR','Bilateral_CFAR')
algo_combobox.current(0)
algo_combobox.grid(row=5,column=3,sticky=tk.N)

## Entry for target area.
tar_win_var = tk.IntVar()
tar_win_entry = ttk.Entry(win,width=12,textvariable=tar_win_var)
tar_win_entry.grid(row=6,column=3,sticky=tk.N)

## Entry for Guard area.
guard_win_var = tk.IntVar()
guard_win_entry = ttk.Entry(win,width=12,textvariable=guard_win_var)
guard_win_entry.grid(row=7,column=3,sticky=tk.N)

## Entry for background area.
background_win_var = tk.IntVar()
background_win_entry = ttk.Entry(win,width=12,textvariable=background_win_var)
background_win_entry.grid(row=8,column=3,sticky=tk.N)

## Choose channel
channel_var = tk.StringVar()
channel_combobox = ttk.Combobox(win, width=15, textvariable=channel_var, state='readonly')
channel_combobox['values'] = ('Select Channel','VH','VV')
channel_combobox.current(0)
channel_combobox.grid(row=9,column=3,sticky=tk.N)

## Entry for pfa
pfa_win_var = tk.IntVar()
pfa_win_entry = ttk.Entry(win,width=12,textvariable=pfa_win_var)
pfa_win_entry.grid(row=10,column=3,sticky=tk.N)

##Submit button
submit_label = ttk.Button(win, text="Start Ship Detection", command=lambda:startDetection())
submit_label.grid(row=11, column=2)

win.mainloop()
