import React, { memo } from "react"
import { Container, Row, Col, Button, Ratio } from "react-bootstrap";
import UploadComponent from "../structural/UploadComponent";
 
import vid from '../../assets/musicvid.mp4'



function LandingPage () {
    return <Container fluid className="d-flex flex-column align-items-center justify-content-center" style={{ minHeight: "100vh" }} >


        <div style={{ width: "100vw", marginLeft: "calc(50% - 50vw)" }}>
            <video style={{width: "100%", objectFit: "cover", height: "25rem"}} autoPlay muted loop>
                <source src={vid} type="video/mp4" />
                Your browser does not support the video tag.
            </video>
        </div>

        <h1 className="text-center display-1 my-5">Sheet Diffusions</h1>
        <h2 className="text-center my-1">Get your favorites all on sheet!</h2>
    
        <p style={{maxWidth: "30rem"}} className="text-center my-5">Have you ever heard a tune that you loved so much you wanted to play it? Have you also faced the roadblock of not being able to find said sheet music? Well look no further, we will create the sheet music for you!</p>    

        <UploadComponent />
    </Container>
}

export default memo(LandingPage);
