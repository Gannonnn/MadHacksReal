import React, { useState, useRef } from "react";
import { Card, Form, Button, Image, Container } from "react-bootstrap";
import Loader from "./Loader";
import { Midi } from "@tonejs/midi";
import * as Tone from "tone";

function UploadComponent(props) {

    const [file, setFile] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [inputKey, setInputKey] = useState(Date.now()); 
    const [errorOccured, setErrorOccured] = useState([false, ""]);
    const [imgUrl, setImageUrl] = useState("");
    const tempoRef = useRef(-1); 
    const [playing, setPlaying] = useState(false);

    const handleFileChange = (event) => {
        setFile(event.target.files[0]); 
    };

    const handlePlayAudio = async () => {
        setPlaying(true);
        setIsLoading(true);
        
        try {
            const response = await fetch("http://127.0.0.1:5000/midi", {
                method: "GET",
            });
            const arrayBuffer = await response.arrayBuffer();

            const midi = new Midi(arrayBuffer);
            console.log("MIDI object:", midi);

            await Tone.start();

            const synth = new Tone.PolySynth(Tone.Synth).toDestination();
            const startTime = Tone.now();

            midi.tracks.forEach(track => {
                track.notes.forEach(note => {
                    synth.triggerAttackRelease(
                        note.name,
                        note.duration,
                        startTime + note.time,
                        note.velocity
                    );
                });
            });

            setInputKey(Date.now());
            setIsLoading(false);
            setPlaying(false);
        } catch (error) {
            alert(`Audio play failed: ${error}`);
            console.error(error);
            setFile(null);
            setIsLoading(false);
            setInputKey(Date.now());
            setPlaying(false);
            return;
        }
    };

    const handleClear = async () => {
        setFile(null);
        setErrorOccured([false, ""])
        setInputKey(Date.now());
        setImageUrl("");
        fetch("http://127.0.0.1:5000/delete", {
            method: "POST",
        });        
    }
    
    const handleUpload = async () => {
        setIsLoading(true);
        if (!file) {
            setIsLoading(false);
            setErrorOccured([true, "You must upload a file to generate sheet music for!"]);
            return;
        }

        if (file.name.slice(-3) !== "mp3") {
            setIsLoading(false);
            setErrorOccured([true, "Files must be of the type .mp3"]);
            return;
        }

        console.log(file)
        const formData = new FormData();
        formData.append("file", file);
        
        try {
            const response = await fetch("http://127.0.0.1:5000/upload", {
                method: "POST",
                body: formData,
            });

            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            
            setImageUrl(url);
            console.log(url);

            setInputKey(Date.now());
            setIsLoading(false);
        } catch (error) {
            alert(`Upload failed: ${error}`);
            console.log(error);
            setFile(null);
            setIsLoading(false);
            setInputKey(Date.now());
            return;
        }
    };

  return (
    <Container>
    <Card className="bg-body-tertiary">
        <Card.Body>
            <Form.Group controlId="formFileLg" className="mb-3" style={{ textAlign: "center" }}>
                <div style={{ marginBottom: "1rem" }}>
                    <Form.Label>Audio File</Form.Label>
                    <Form.Control
                        style={{ maxWidth: "36rem", margin: "0 auto" }}
                        key={inputKey}
                        type="file"
                        size="md"
                        onChange={handleFileChange}
                    />
                </div>

                <div style={{marginTop: "1rem"}}>
                    {isLoading ? <Loader /> : <></>}
                    <Button variant="success" onClick={handleUpload} disabled={!file}>
                        Upload
                    </Button>
                    <span className="mx-2"></span>
                    <Button onClick={handleClear} disabled={!file} variant="outline-danger">
                        Clear
                    </Button>      
                {errorOccured[0] ? <div><span style={{color: "red"}}>{errorOccured[1]}</span></div> : <></>}
                </div>
            </Form.Group>
        </Card.Body>
        <Card.Footer>We do not train on your data, and we will delete your recording after we are done processing it.</Card.Footer>
        </Card>
        { imgUrl ? 
        <Container>
            <Card>
                <Card.Body className="d-flex justify-content-center">
                    <Image width="50%" src={imgUrl} />
                </Card.Body>
                <Button disabled={playing} onClick={handlePlayAudio} variant="primary">Hear the audio! <i className="bi bi-volume-up fs-6"></i></Button>
            </Card>
        </Container>
        :
        <></>
        }
    </Container>
  );
}

export default UploadComponent;
