# Song Recognition Algorithm
## 1. Introduction
Approximately two months ago I stumbled onto [this](https://www.youtube.com/watch?v=a0CVCcb0RJM) youtube video about recreating Shazam. After watching the video I was quite surprised by the apparent simplicity of the algorithm. Of course I realized this is a very broad overview of what the algorithm does. So after looking at the comments I found a comment from Ron Van Rijn from whom the original video creator had taken some inspiration. I have then found a [blog](https://www.royvanrijn.com/blog/2010/06/creating-shazam-in-java/) and a [youtube video](https://www.youtube.com/watch?v=T4PJoAh4X1g) from the commenter which have then served as the main source for this project. So the question of this project is how to recognize that two recordings are of the same song. 
## 2. Data
Before starting to work on the actual algorithm the problem of where to find a sufficient amount of songs had to be solved. A convenient way to download them was via youtube using a library called `yt_dl`. Still a list of songs to download was required. For this I the inspiration was taken from the main source where the Rolling Stone Top 500 Songs was used. To use this in the program I found a [repository](https://gist.githubusercontent.com/keune/0de5c7fb669f7b682874/raw/4aabd7282ee6b58ff886af50489cbcc6c4bd1faf/RollingStone%2520Top%2520500%2520Song) containing a json file with the list. For this to be done efficiently I used multiprocessing so that multiple songs can get downloaded at once and stored in an sqlite database. 
## 3. Algorithm
### 3.1 Sound
To understand the algorithm we have to first understand that sound is a waveform which gets stored digitally by sampling its amplitude at a chosen sampling rate such as $44.1\text{kHz}$. When we read the data into a program in python we usually get a one dimensional numpy array:

<div style="text-align: center; font-size: 15px; font-family: monospace;">
<pre>
[0.00101867 0.00158976 0.00167912 ... 0.00011148 0.00012591 0.00010241]
</pre>
</div>

Each element of this array presents an amplitude of the waveform at a point in time. So if we have a recording of length $1\text{s}$ at $44.1\text{kHz}$ sampling rate this array would be of length $44100$. We can also visualize this signal on a plot of amplitude to relation to time: 

<div style="text-align: center;">
  <img src="images/sound.png" alt="Image of a sound waveform" width="400"/>
</div>

### 3.2 Fourier Transform and Spectrograms
The underlying idea of Fourier Transform is that all sound waveforms are a sum of sine waves:

$s(t) = \sum_{n} C_n \sin\big(2\pi f_n t + \phi_n\big)$

With Fourier Transform we can then extract the frequencies $f_n$ and their respective coefficients $C_n$. For example if we take a signal $s(t) = \sin(2\pi t) + 0.5\sin(2\pi 3 t)$ this is the output of the Discrete Fourier Transform:

<div style="text-align: center;">
  <img src="images/ft.png" alt="Fourier transform of sine waves" width="700"/>
</div>

Our signal is not changing frequencies through time which is not true in sound. So an interesting thing to draw is a spectrogram which shows how magnitudes of different frequencies changes through time. To do this the signal gets split into an overlapping windows. For example if we have a $2$s signal and $1$s windows we can use the following intervals $0$-$1$s, $0.5$-$1.5$s, $1$-$2$s. An interesting example of a spectrogram is from a song Equation by Aphex Twins which has a face drawn in the last $30$s:

<div style="text-align: center;">
  <img src="images/spectrogram.png" alt="Spectrogram of Aphex Twin song Equation" width="500"/>
</div>

### 3.3 Fingerprinting
#### 3.3.1 Creating a Fingerprint
So the goal is to have a database of songs to which we can link a new recording of a song. After drawing a spectrogram of a song the problem gets changed into how do we recognize that the two images are the same. First thing we do is limit ourselves to frequencies below $6000\text{Hz}$ which according to [wikipedia](https://en.wikipedia.org/wiki/Hearing_range) and [gear4music](https://www.gear4music.com/blog/audio-frequency-range/) are most important for human hearing sensitivity.
What we would like to do next is design numbers that describes a unique feature of the spectogram that can then be looked up in other words hashes. To do this we seperate the frequencies into $8$ channels and for every time window we find a maximum magnitude within each channel. Then we break down each channel into $8$ subchannels and check which subchannel the maximum lies in. If the maximum lies in the subchannel then its value is $1$ otherwise it is $0$. This gives us $8$ binary values for every channel which can then be formated into a $64$ bit integer. 
This is done on every song that we have and is then stored in the database alongside its timestamp and of course indexed for a quick lookup.
#### 3.3.2 Fingerprint Matching
Now that the hashes are stored in the database a lookup system has to be established. As we process a recording we construct windows of the desired length then find its fingerprint with an equivalent process described before. After multiple hashes are found that are in the same song in the database we look at the time difference between the two. If the time difference is small enough, that is considered a match. As we progress through the recording we rank the songs by matches and ideally get a song with significantly more matches than the others. 
## 4. Results
This algorithm has some mixed results. For cuts of the songs taken from the original recordings it works rather well and after processing the entire recording a vast majority of the matches are correct. However, for actual recordings through a microphone results vary quite a lot and a correct match is somewhat rare. The reason for this could result from the best match windows being very short and potentially having a lot of hashes found close to each other therefore resulting in a false match. I have attempted to fix this by making variations of the hash algorithm and changing the match metric but this was rather unsuccesfull. I have also found a [more complex algorithm](https://drive.google.com/file/d/1ahyCTXBAZiuni6RTzHzLoOwwfTRFaU-C/view) for matching the songs which incorporates the time difference into the hash with target zones. This addresses the issues encountered so the next step for a better success rate is to implement this modified hash.  
## 5. Running the code
To run the code make sure Python is installed and install all the dependencies:
`pip install -r requirements.txt`
Then you first have to construct the database of songs. This will download ~$500$ songs from youtube which is around $3$GB and create the database with song information and fingerprints. You can run this with:
`python main.py store`
After you have done this you can run `main.py` with no arguments which will run the song recognition algorithm on some provided examples. If you wish to run it on your own recording you can provide the path to file as an argument:
`python main.py filepath`
Alternatively you can also run the file `website.py` to put up a website on a [local address](http://0.0.0.0:8000) where a live recording will get classified after pressing `start recording` and the output will be printed in terminal on the server side. The website and the server are unfinished as the algorithm did not perform well for actual recordings.  