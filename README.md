# MadHacks 2025 - Sheet Diffusion
Charles Ding, Marko Kupresanin, Gannon Mefford, Victoria Yang

## Inspiration
We were inspired by the lack of free software that transcribes audio files into proper sheet music. This was important to us because musicians who are early in their education/training may have trouble finding, self-curating, or understanding how to create the sheet music for their favorite songs. By creating this app utilizing Fish Diffusion we enable young musicians to be better connected with music in general.

## What it does
Our React.JS frontend allows for ```.mp3``` files to be directly uploaded to our backend server which implements the former Fish Diffusion model that cleverly extracts the pitches observed in the recording. We use this data from Fish to understand the [frequency of sounds as musical notes](https://en.wikipedia.org/wiki/Piano_key_frequencies), and process the spacing between notes with OpenCV. From there, we parse all the rests and frequency data to utilize MuScore's Python SDK which creates high quality renditions of sheet music with the necessary time signatures, clef, and accidentals.

Finally, we utilize a React.js library to take the MIDI file we generated from the transcribed audio file and allow the end-user to hear what the sheet music sounds like, just as an added convenience so they can make sure to match the correct key and all!

## How we built it
We used the [Fish Diffusion](https://github.com/fishaudio/fish-diffusion/) framework developed by fish.audio researchers to detect pitch changes and successfully isolate voices and musical instruments as needed for sheet music. 

We also used [Flask](https://flask.palletsprojects.com/en/stable/) for our locally hosted backend which allowed us to continue to use the Python implementation of Fish Diffusion and the MuseScore SDK to crunch the numbers and generate sheet music

Our frontend was created with React.js built with [Vite](https://vite.dev/guide/), it is styled with React Bootstrap and follows WCAG AA standards for accessibility.

## Challenges we ran into
We ran into challenges understanding the Fish Diffusion framework as it was not the typical TTS/STT offering fish.audio is currently presenting, but we continued to iterate and read many, many documentation pages from Fish and their dependencies which allowed us to eventually come up with clever solutions, such as using OpenCV to detect start and end of notes.
We also ran into issues using MuseScore's Python SDK, music21. Primarily due to the overly complicated wiki page which spread out resources across 20+ chapters, making error debugging difficult. Additionally, we ran into issued with MuseScore's entire website being inaccessible and were met with many Cloudflare error screens. This was coupled with the music21 SDK requiring us to specify very specific paths on our machines just for the image generation to work, which was not documented whatsoever in their wiki. This also prevented easily hosting the backend due to the need for the MuseScore app which **has** to be downloaded through a third party marketplace.

## Accomplishments that we're proud of
We are very proud of all our team members because while it was difficult brainstorming our initial Hackathon ideas and even starting out with our final decision for this submission, we were able to pull through as a team to deliver a fully finished product based on our original intentions. The final product exceeded our expectations during our beginning struggles.
We are also proud of being able to come up with an idea that was able to come to fruition.

## What we learned
We learned that there are a lot of different ways to process music and that having a truly accurate model will require some more advanced math than we could understand during these 24 hours. 

## What's next for Sheet Diffusions
We hope to improve even further on fish.audio's Fish Diffusion by either incorporating some of our own agentic AI workflows to really contextualize the audio so we can better understand the eloquence of music so the sheet music is more accurate. We also would possibly look into VexFlow as an alternative to MuseScore due to their open source nature and less reliance on a 3rd party marketplace to install the tool needed for generating our sheet music.
