import React, { memo } from "react"
import { Container, Row, Col, Button } from "react-bootstrap";
import UploadComponent from "../structural/UploadComponent";
 
function LandingPage () {
    return <Container fluid className="d-flex flex-column align-items-center justify-content-center" style={{ minHeight: "100vh" }} >
        <h1 className="text-center display-1 my-5">Sheet Diffusions</h1>
        <h2 className="text-center my-1">Get your favorites all on sheet!</h2>
    
        <p style={{maxWidth: "30rem"}} className="text-center my-5">Have you ever heard a tune that you loved so much you wanted to play it? Have you also faced the roadblock of not being able to find said sheet music? Well look no further, we will create the sheet music for you!</p>    

        <UploadComponent />
    </Container>
}

export default memo(LandingPage);
