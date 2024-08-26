import { Box, Container, Grid, GridItem, Show } from '@chakra-ui/react';
import NavBar from './components/NavBar';
import ChapterGrid from './components/ChapterGrid';
import ChapterList from './components/ChapterList';

function App() {
  return (
    <Box>
      <NavBar/>
      <Container maxW='1200px' px='10px' mt='72px'>
        <Grid
          templateAreas={{
            base: `'main'`,
            lg: `'aside main'`,
          }}
          templateColumns={{
            base: '1fr',
            lg: '350px 1fr',
          }}
          gap={4}
          >
          <Show above='lg'>
            <GridItem area='aside' px={3} py={1} bg='gray.100' >
              <Box
                pt={3}
                position='sticky'
                top='72px'
                maxH='calc(100vh - 72px)' // Limit the max height based on viewport
                overflowY='auto' // Enable scrolling inside ChapterList if content overflows
              >
                <ChapterList/>
              </Box>
            </GridItem>
          </Show>
          <GridItem area='main'>
            <ChapterGrid/>
          </GridItem>
        </Grid>
      </Container>
    </Box>
  );
}

export default App;