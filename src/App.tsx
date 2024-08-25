import { Container, Grid, GridItem, Show } from '@chakra-ui/react';
import NavBar from './components/NavBar';
import ChapterGrid from './components/ChapterGrid';
import ChapterList from './components/ChapterList';

function App() {
  return (
    <Container maxW='1200px' px='10px'>
      <Grid
        templateAreas={{
          base: `'nav' 'main'`,
          lg: `'nav nav' 'aside main'`,
        }}
        >
        <GridItem area='nav'><NavBar/></GridItem>
        <Show above='lg'>
          <GridItem area='aside' px={3} mr='10px' bg='gray.100' maxW="350px">
            <ChapterList/>
          </GridItem>
        </Show>
        <GridItem area='main'><ChapterGrid/></GridItem>
      </Grid>
    </Container>
  );
}

export default App;