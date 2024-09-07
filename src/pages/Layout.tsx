import { Box, Container, Grid, GridItem, Show } from '@chakra-ui/react'
import NavBar from '../components/NavBar'
import { Outlet } from 'react-router-dom'
import BookList from '../components/BookList'
import ScrollToTop from '../components/ScrollToTop'

const Layout = () => {
  return (
    <>
      <NavBar/>
      <Container maxW='1200px' px='10px' mt='72px'>
        <Grid
          templateAreas={{ base: `'main'`, lg: `'aside main'` }}
          templateColumns={{ base: '1fr', lg: '320px 1fr' }}
          gap={4}
          >
          <Show above='lg'>
            <GridItem area='aside' p={1} bg='gray.100' >
              <Box
                py={3}
                position='sticky'
                top='72px'
                maxH='calc(100vh - 72px)'
                overflowY='auto'
              >
                <BookList/>
              </Box>
            </GridItem>
          </Show>
          <GridItem area='main' minWidth="0">
            <ScrollToTop/>
            <Outlet/>
          </GridItem>
        </Grid>
      </Container>
    </>
  )
}

export default Layout

// minWidth="0":
// to ensure it doesn’t force the content to expand beyond the screen width.