# Sustain Pedal for Voice
Hold "L" to sustain your voice. You can sing chords without outside help in real time.  

Whenever the sustain pedal is down AND your voice is stable in pitch, the program records your voice (timbre + pitch) and sustains a resynthesis of it.  

The sustained notes track the expression/volume of your voice, so if you crescendo, "your choir" also will.  

`STABILITY_THRESHOLD`: if too high, it's hard to trigger a note. If too low, non-notes may be sustained.  
`MAIN_VOICE_THRU`: if `False`, the program will not output your main voice - there will only be accompaniment (your previous sustained notes).   
`N_HARMONICS`: How many harmonics your voice is represented.  
`EXPRESSION_SWIFT`: Lower values smooth the expression curve, creating a slow-release effect.  
`DO_PROFILE`: whether to profile runtime efficiency.  
`PEDAL_DELAY`: under the current implementation, it should be larger than x and smaller than y: if you hold a key on your keyboard, a char is immediately sent, followed by a delay of y, and then a stream of chars with interval x.  
`WRITE_FILE`: write to a wav file. If `None`, does not write.  
`REALTIME_FEEDBACK`: If `False`, it only writes the file, and there is no audio out from the speakers.  

## Demo
Demo 1: https://youtu.be/PxjjnCc7VJw
Demo 2: https://youtu.be/9ino2go_F9k

## Discussion
My imagination is that with this tool, people like Jacob Collier can draft musical ideas more easily.  

Some side discoveries. When the program detects a new note that is closer in pitch to a previous note than `OVERWRITE_MIN_PITCH_DIFF`, it replaces the previous note with the new note. Its original purpose is to support correction of old notes. However, as I was making demo 2, I realized I could use this feature to release sustained notes by singing anything close to it. Many parts of demo 2 were thus created with that technique.  
A larger `OVERWRITE_MIN_PITCH_DIFF` will also makes the program fool-proof, because small intervals will not be producable, prohibiting dissonance.  
