import React, { useState, useRef } from "react";
import { Card, Form, Button } from "react-bootstrap";
import Loader from "./Loader";


function UploadComponent(props) {

    const [file, setFile] = useState(null);
    const [isLoading, setIsLoading] = useState(null);
    const [inputKey, setInputKey] = useState(Date.now()); 
    const tempoRef = useRef(-1); 

    const handleFileChange = (event) => {
        setFile(event.target.files[0]); 
    };

    const handleClear = (event) => {
        setFile(null);
        setInputKey(Date.now());
    }
    
    const handleUpload = async () => {
        setIsLoading(true);
        if (!file) return alert("Please select a file first!");
        console.log(file)
        const formData = new FormData();
        formData.append("file", file);
        
        try {
            const response = await fetch("https://madhacksreal.onrender.com/upload", {
                method: "POST",
                body: formData,
            });

            const data = await response.json();
            console.log(data);
            setIsLoading(false);
        } catch (error) {
            alert(`Upload failed: ${error}`);
            setIsLoading(false);
            setInputKey(Date.now());
            return;
        }
    };

  return (
    <Card className="bg-body-tertiary">
        <Card.Body>
            <Form.Group controlId="formFileLg" className="mb-3" style={{ textAlign: "center" }}>
                <div style={{ marginBottom: "1rem" }}>
                    <Form.Label>Large file input example</Form.Label>
                    <Form.Control
                        style={{ maxWidth: "36rem", margin: "0 auto" }}
                        key={inputKey}
                        type="file"
                        size="md"
                        onChange={handleFileChange}
                    />
                </div>

                <div>
                    <Form.Label>Tempo</Form.Label>
                    <Form.Control
                        style={{ maxWidth: "6rem", margin: "0 auto" }}
                        placeholder="128"
                        disabled={true}
                        key={inputKey + 1}
                        type="number"
                        size="md"
                        ref={tempoRef}
                    />
                </div>
                <div style={{marginTop: "3rem"}}>
                    {isLoading ? <Loader /> : <></>}
                    <Button variant="success" onClick={handleUpload} disabled={!file}>
                        Upload
                    </Button>
                    <span className="mx-2"></span>
                    <Button onClick={handleClear} disabled={!file} variant="outline-danger">
                        Clear
                    </Button>      
                </div>
            </Form.Group>

            {/* <Form.Group controlId="formFileLg" className="mb-3">
                <Form.Label>Large file input example</Form.Label>
                <Form.Control style={{maxWidth: "36rem"}} key={inputKey} type="file" size="md" onChange={handleFileChange} />

                <Form.Label>Tempo</Form.Label>
                <Form.Control style={{maxWidth: "6rem"}} placeholder="128" disabled={true} key={inputKey+1} type="number" size="md" ref={tempoRef}/>
            </Form.Group> */}
        </Card.Body>
    </Card>
  );
}

export default UploadComponent;
