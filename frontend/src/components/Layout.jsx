import React, { useEffect, useState, useContext } from "react";
import { Container, Nav, Navbar, NavDropdown, Button } from "react-bootstrap";
import { Link, Outlet } from "react-router";

import logo from '../assets/madhacks25logo.png'

function Layout(props) {
    

    return (
        <div className="bg-light-subtle pb-5">
            <Navbar collapseOnSelect expand="sm" className="bg-dark-subtle">
                <Container>
                    <Navbar.Brand as={Link} to="/" className="">
                        <img
                            alt="sheet diffusions logo with a eigth note and usb symbol"
                            src={logo}
                            width="30"
                            height="30"
                            className="d-inline-block align-top"
                        />{' '}
                        Sheet Diffusions
                    </Navbar.Brand>
                    <Navbar.Toggle aria-controls="responsive-navbar-nav" />
                    <Navbar.Collapse id="responsive-navbar-nav" className="justify-content-end">
                    <Nav>
                        <Button variant="outline-primary" as={Link} to="https://madhacks-fall-2025.devpost.com/" target="_blank" rel="noopener noreferrer" >DevPost </Button>
                    </Nav>
                    </Navbar.Collapse>
                </Container>
            </Navbar>
            
            <div style={{ margin: "1rem" }}>
                <Outlet />
            </div>

        </div>
    );
}

export default Layout;